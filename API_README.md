# RAG API Documentation

## Overview
This API provides endpoints for uploading files and querying a RAG (Retrieval-Augmented Generation) system using LlamaIndex and Qdrant vector store.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant (if not already running):
```bash
# For development/testing
docker run -p 6333:6333 qdrant/qdrant

# For production (runs in background)
docker run -d -p 6333:6333 qdrant/qdrant
```

Or set environment variables for remote Qdrant:
```bash
export QDRANT_URL=http://your-qdrant-host:6333
export QDRANT_API_KEY=your-api-key  # Optional
```

## Running the API

Start the API server:
```bash
python api.py
```

Or use uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: GET
- **Description**: Get API information and available endpoints

### 2. Health Check
- **URL**: `/health`
- **Method**: GET
- **Description**: Check API health status

### 3. Upload File
- **URL**: `/upload`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file` (required): PDF file to upload
  - `embed_model` (optional, default: "qwen"): Embedding model to use
    - Options: "qwen", "bge", "m3", "multilingual"

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "embed_model=qwen"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/upload"
files = {"file": open("document.pdf", "rb")}
data = {"embed_model": "qwen"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response:**
```json
{
  "message": "File uploaded and indexed successfully",
  "filename": "document.pdf",
  "nodes_count": 42
}
```

### 4. Query Documents
- **URL**: `/query`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body**:
  - `query` (required): Query text to search for
  - `embed_model` (optional, default: "qwen"): Embedding model to use
  - `top_k` (optional, default: 3): Number of top results to return (1-100)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is fine-tuning?",
    "embed_model": "qwen",
    "top_k": 5
  }'
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/query"
payload = {
    "query": "What is fine-tuning?",
    "embed_model": "qwen",
    "top_k": 5
}

response = requests.post(url, json=payload)
print(response.json())
```

**Response:**
```json
{
  "results": [
    {
      "score": 0.85,
      "content": "Fine-tuning is a process...",
      "metadata": {"page": 1, "file_name": "document.pdf"}
    },
    {
      "score": 0.78,
      "content": "Another relevant passage...",
      "metadata": {"page": 3, "file_name": "document.pdf"}
    }
  ],
  "total_results": 2
}
```

## Embedding Models

The API supports the following embedding models:

| Model Name | HuggingFace Model ID | Description |
|------------|---------------------|-------------|
| qwen (default) | Qwen/Qwen3-Embedding-0.6B | Qwen multilingual embedding model |
| bge | BAAI/bge-small-en | BGE small English model |
| m3 | BAAI/bge-m3 | BGE M3 multilingual model |
| multilingual | intfloat/multilingual-e5-base | Multilingual E5 base model |

## Interactive API Documentation

Once the server is running, you can access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (e.g., invalid file type, empty query)
- `500`: Server error (e.g., processing error)

Error responses include a `detail` field with the error message:
```json
{
  "detail": "Only PDF files are supported"
}
```

## Notes

- Only PDF files are currently supported for upload
- Files are temporarily stored during processing and then deleted
- The vector store persists data in Qdrant
- Embedding models are loaded on-demand and cached
