from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from google.cloud import storage, aiplatform
from google.oauth2 import service_account
from langchain.docstore.document import Document
import os, tempfile, uuid, shutil, pymupdf, logging
from langchain_google_vertexai import VertexAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GCS_BUCKET = "pdf-chat-app-new"
BUCKET_NAME = "pdf-chat-app-new"
FAISS_INDEX_PATH = "faiss_indices"

PROJECT_ID = "pdfchatapp-459613"
LOCATION = "us-central1"

credentials = service_account.Credentials.from_service_account_file("pdfchatapp.json")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

try:
    llm = VertexAI(
        model_name="gemini-2.0-flash-lite",
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials,
        max_output_tokens=1000,
        temperature=0.1,
    )
    logger.info("Successfully initialized Vertex AI model: gemini-2.0-flash-lite")
except Exception as e:
    logger.error(f"Error loading the language model: {str(e)}", exc_info=True)
    llm = None
    raise RuntimeError(f"Failed to initialize language model: {e}")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert financial document analysis tool.
    Extract the exact value or information corresponding to the financial metric or query specified by the user.
    Provide a concise answer in 3–4 lines, citing the page number if available.
    You are allowed to perform calculations (such as difference, percentage change, etc.) using numbers found in the chunks.
    If a query asks for a percentage increase, and both old and new values are in the chunks, compute it as:
      ((new - old) / old) * 100
    Do NOT fabricate data not present in the chunks.
    Do NOT generate follow-up questions, repeat the query.
    """),
    ("human", "Document Chunks:\n{context}\n\nQuery: {query}\n\nAnswer:")
])

chain = prompt_template | llm | StrOutputParser()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def upload_to_gcs(file_path, destination_blob_name):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    return f"gs://{BUCKET_NAME}/{destination_blob_name}"

def download_from_gcs(source_blob_name, destination_file_path):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)

def load_pdf_with_pymupdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    texts = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        texts.append(Document(page_content=text, metadata={"page": page_num + 1}))
    doc.close()
    return texts

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_id = str(uuid.uuid4())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf_path = temp_pdf.name
            temp_pdf.write(await file.read())

        upload_to_gcs(temp_pdf_path, f"pdfs/{pdf_id}.pdf")

        documents = load_pdf_with_pymupdf(temp_pdf_path)
        texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

        vector_store = FAISS.from_documents(texts, embedding_model)
        temp_index_dir = tempfile.mkdtemp()
        temp_index_path = os.path.join(temp_index_dir, f"{pdf_id}_faiss")
        vector_store.save_local(temp_index_path)

        upload_to_gcs(f"{temp_index_path}/index.faiss", f"{FAISS_INDEX_PATH}/{pdf_id}/index.faiss")
        upload_to_gcs(f"{temp_index_path}/index.pkl", f"{FAISS_INDEX_PATH}/{pdf_id}/index.pkl")

        os.remove(temp_pdf_path)
        shutil.rmtree(temp_index_dir, ignore_errors=True)

        return JSONResponse({"message": "PDF processed", "pdf_id": pdf_id})

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    query: str

@app.post("/query/{pdf_id}")
async def query_pdf(pdf_id: str, req: QueryRequest):
    if not llm:
        logger.error("Language model not initialized.")
        raise HTTPException(status_code=500, detail="Language model not initialized.")
    try:
        query = req.query.lower()
        logger.info(f"Processing query: {query} for PDF ID: {pdf_id}")

        temp_index_dir = tempfile.mkdtemp()
        temp_index_path = os.path.join(temp_index_dir, f"{pdf_id}_faiss")
        os.makedirs(temp_index_path, exist_ok=True)
        logger.info(f"Created temporary directory: {temp_index_path}")

        logger.info("Downloading FAISS index from GCS...")
        download_from_gcs(f"{FAISS_INDEX_PATH}/{pdf_id}/index.faiss", f"{temp_index_path}/index.faiss")
        download_from_gcs(f"{FAISS_INDEX_PATH}/{pdf_id}/index.pkl", f"{temp_index_path}/index.pkl")
        logger.info("FAISS index downloaded successfully.")

        logger.info("Loading FAISS vector store...")
        vector_store = FAISS.load_local(temp_index_path, embedding_model, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded successfully.")

        logger.info("Performing similarity search...")
        results = vector_store.similarity_search(query, k=10)
        logger.info(f"Similarity search returned {len(results)} results.")

        key_phrases = []
        if "gross revenue" in query:
            key_phrases = ["gross revenue", "sale of products", "revenue"]
        elif "where" in query and "company" in query:
            key_phrases = ["registered office", "address", "based", "location"]
        else:
            key_phrases = query.split()[:3]
        logger.info(f"Key phrases for filtering: {key_phrases}")

        relevant_chunks = [r for r in results if any(phrase in r.page_content.lower() for phrase in key_phrases)]
        logger.info(f"Found {len(relevant_chunks)} relevant chunks after filtering.")

        if relevant_chunks:
            context = "\n\n".join([f"Chunk {i + 1} (Page {r.metadata['page']}):\n{r.page_content}" for i, r in enumerate(relevant_chunks)])
        else:
            context = "\n\n".join([f"Chunk {i + 1} (Page {r.metadata['page']}):\n{r.page_content}" for i, r in enumerate(results)])
        logger.info("Context prepared for the language model.")

        logger.info("Invoking the language model chain...")
        try:
            response = chain.invoke({"query": query, "context": context})
            logger.info("Language model chain invoked successfully.")
        except Exception as invoke_error:
            logger.error(f"Failed to invoke language model chain: {str(invoke_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Language model invocation failed: {str(invoke_error)}")

        answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.strip()
        logger.info(f"Processed answer: {answer}")

        shutil.rmtree(temp_index_dir, ignore_errors=True)
        logger.info("Temporary directory cleaned up.")

        return JSONResponse({
            "query": query,
            "raw_chunks": [{"content": r.page_content, "page": r.metadata["page"]} for r in results],
            "summary": answer
        })
    except Exception as e:
        logger.error(f"Error in query_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))