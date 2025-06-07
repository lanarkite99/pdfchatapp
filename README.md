# **PDF Chat App for Financial Document Querying**

A **web application** that allows users to upload **financial PDFs** and query their content (e.g., revenue, income) using natural language, powered by **AI and vector search**. Built with **FastAPI (backend)** and **React (frontend)**, the app leverages **Vertex AI (gemini-2.0-flash-lite)** and a locally hosted **LLaMA 3.2 1B-Instruct** model for answers, with **embeddings** stored in **FAISS**and data managed in **Google Cloud Storage (GCS)**. Deployed on **Cloud Run (us-central1)**.


## **Features**
- 📄 Upload financial PDFs and query metrics (e.g., "What is the Q1 2025 revenue?") via a user-friendly React interface.
- ❓ Extract text from PDFs using **PyMuPDF** and split into chunks with **LangChain** for efficient processing.
-  Generate **embeddings** with **HuggingFace (sentence-transformers/all-mpnet-base-v2)** and perform fast similarity search using FAISS.
- 🔍 Answer queries using Vertex AI (Gemini 2.0 Flash) for scalable inference and a locally hosted LLaMA 3.2 1B-Instruct model (weights and tokenizer on GCS) for custom deployment.
- ☁️ Store PDFs and FAISS indices in GCS (pdf-chat-app-new) for seamless scalability.
- 🚀 Deployed on Cloud Run (asia-south1) with Docker for production-ready performance.

## **Tech Stack**
- **Languages**: Python, JavaScript, Bash
- **Backend**: FastAPI
- **Frontend**: React, Axios, Tailwind CSS, Clipboard.js, Babel
- **AI/ML**: Vertex AI (gemini-2.0-flash-lite), LLaMA 3.2 1B-Instruct, LangChain, FAISS, HuggingFace Transformers
- **Cloud**: Google Cloud Platform (Cloud Run, GCS, Vertex AI, Artifact Registry)
- **Other Tools**: PyMuPDF, Docker, Git

## **Installation**

**Prerequisites**
- Python 3.9+
- Node.js 18+
- Google Cloud SDK (gcloud)
- Docker
- A Google Cloud project with billing enabled
- Vertex AI API enabled and a service account key (pdfchatapp.json)
- Access to LLaMA 3.2 1B-Instruct model weights (research use only)

## **Steps**
1. **Clone the Repository:**
   ```
   git clone https://github.com/lanarkite99/pdfchatapp.git  
   cd pdfchatapp
   ```
   
2. **Set Up Python Environment:**
   ```
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   pip install -r requirements.txt  
   ```
   
3. **Set Up Environment Variables:**
    Create a .env file in the root directory:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/pdfchatapp.json  
   PROJECT_ID=your-proj-name 
   LOCATION=us-central1  
   BUCKET_NAME=pdf-chat-app-new
   ```
   Place your Google Cloud service account key (pdfchatapp.json) in the specified path.
   
5. **Download LLaMA Model (Optional):**
   If using the locally hosted LLaMA 3.2 1B-Instruct model, download its weights and tokenizer (research use only) and upload them to GCS (gs://pdf-chat-app-new/llama-     weights/).
   
7. **Run Locally:**
   Start the FastAPI backend:
   ```uvicorn main:app --reload --host 0.0.0.0 --port 8000```
   Open index.html in a browser to access the frontend locally (or serve it via a static server).

## **Usage**
1. **Upload a PDF:**
    Use the React frontend to upload a financial PDF (e.g., annual report).
    The backend processes the PDF, extracts text, and stores it in GCS.
2. **Query the PDF:**
    Enter a query (e.g., "What is the total income for Q1 2025?").
    The app uses FAISS to find relevant chunks and generates an answer using Vertex AI or LLaMA.
3. **View Results:**
    The frontend displays the query, raw chunks, and a summarized answer with a "Copy" button.

## **Deployment to Cloud Run**
  1. **Build the Docker Image:**
     ```gcloud builds submit --tag us-central1-docker.pkg.dev/your-project-name/pdfchatnew/pdf-chat-app:latest .  ```
  2. **Deploy to Cloud Run:**
     ```
     gcloud run deploy pdf-chat-app \  
      --image us-central1-docker.pkg.dev/your-project-name/pdfchatnew/pdf-chat-app:latest \  
      --region asia-south1 \  
      --platform managed \  
      --memory 16Gi \  
      --cpu 4 \  
      --timeout 900 \  
      --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/app/pdfchatapp.json" \  
      --allow-unauthenticated
  
  3. **Access the Deployed App:**
       After deployment, Cloud Run provides a URL (e.g., https://pdf-chat-app.asia-south1.run.app).
       Update index.html URLs to point to this deployed service.

## **Contributing**
  Contributions are welcome! Please follow these steps:

    1. Fork the repository.
    2. Create a new branch (git checkout -b feature/your-feature).
    3. Commit your changes (git commit -m "Add your feature").
    4. Push to the branch (git push origin feature/your-feature).
    5. Open a pull request.

## **License**
  This project is licensed under the MIT License. See the LICENSE file for details.

## **Acknowledgments**
  Thanks to the creators of LangChain, FAISS, and Google Cloud for their amazing tools.
  Inspired by various PDF chat applications for AI-driven document querying.
   
