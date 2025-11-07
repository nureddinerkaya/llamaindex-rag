# Quick Start Guide for RAG API

## Prerequisites
- Python 3.8+
- Docker (for running Qdrant)

## Step 1: Start Qdrant Vector Database

```bash
# Start Qdrant in Docker
docker run -d -p 6333:6333 qdrant/qdrant
```

## Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## Step 3: Start the API Server

```bash
# Option 1: Using Python directly
python api.py

# Option 2: Using uvicorn with auto-reload (for development)
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

## Step 4: Upload a Document

```bash
# Upload a PDF file to the vector store
curl -X POST "http://localhost:8000/upload" \
  -F "file=@./data/llama2.pdf" \
  -F "embed_model=qwen"
```

## Step 5: Query the Documents

```bash
# Query for information
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is fine-tuning?",
    "embed_model": "qwen",
    "top_k": 3
  }'
```

## Step 6: Explore Interactive Documentation

Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly from these interfaces!

## Using Different Embedding Models

The API supports multiple embedding models:

```bash
# Using BGE model for upload
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "embed_model=bge"

# Using M3 model for query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question",
    "embed_model": "m3",
    "top_k": 5
  }'
```

Available models:
- `qwen` (default) - Qwen/Qwen3-Embedding-0.6B
- `bge` - BAAI/bge-small-en
- `m3` - BAAI/bge-m3
- `multilingual` - intfloat/multilingual-e5-base

## Running Tests

```bash
# Run the test script (make sure API is running first)
python test_api.py

# Or specify a custom PDF file
python test_api.py /path/to/your/document.pdf
```

## Environment Variables

You can configure Qdrant connection using environment variables:

```bash
# For remote Qdrant instance
export QDRANT_URL=http://your-qdrant-host:6333
export QDRANT_API_KEY=your-api-key

# Then start the API
python api.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'llama_index'"
- Make sure you've installed all dependencies: `pip install -r requirements.txt`

### "Connection refused" when uploading/querying
- Make sure Qdrant is running: `docker ps | grep qdrant`
- Start Qdrant if not running: `docker run -d -p 6333:6333 qdrant/qdrant`

### "Only PDF files are supported"
- The current version only supports PDF files
- Make sure your file has a .pdf extension

## Next Steps

For more detailed information, see:
- [API_README.md](API_README.md) - Complete API documentation
- Swagger UI at http://localhost:8000/docs - Interactive API testing
