FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade transformers tokenizers sentencepiece

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY static/ ./static/
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]