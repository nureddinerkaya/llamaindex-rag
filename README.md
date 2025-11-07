# LlamaIndex RAG Application

A Retrieval-Augmented Generation (RAG) system built with LlamaIndex, Qdrant vector store, and FastAPI.

## 🚀 Features

- **REST API**: FastAPI-based REST API for easy integration
- **File Upload**: Upload PDF documents with automatic chunking and embedding
- **Query System**: Semantic search across uploaded documents
- **Multiple Embedding Models**: Support for various embedding models (Qwen, BGE, M3, Multilingual)
- **Vector Store**: Qdrant for efficient similarity search
- **Interactive Documentation**: Swagger UI and ReDoc for API exploration

## 📋 Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide.

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start the API
python api.py
```

### Usage

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "embed_model=qwen"

# Query documents
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is fine-tuning?", "top_k": 3}'
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Step-by-step getting started guide
- **[API_README.md](API_README.md)**: Complete API documentation with examples
- **[example_usage.py](example_usage.py)**: Python code examples
- **[test_api.py](test_api.py)**: API testing script

## 🏗️ Architecture

### Components

- **api.py**: FastAPI application with upload and query endpoints
- **model_ve_db.py**: Embedding model and Qdrant vector store configuration
- **yukleme.py**: Document loading and processing utilities
- **sorgulama.py**: Query processing utilities
- **retriever.py**: Custom retriever implementation

### Workflow

1. **Upload**: PDF → Load → Chunk → Embed → Store in Qdrant
2. **Query**: Question → Embed → Search Qdrant → Return top-k results

## 🎯 API Endpoints

### POST /upload
Upload and index a PDF document.

**Parameters:**
- `file`: PDF file (required)
- `embed_model`: Embedding model (optional, default: "qwen")

### POST /query
Search for relevant information.

**Body:**
```json
{
  "query": "your question",
  "embed_model": "qwen",
  "top_k": 3
}
```

### GET /docs
Interactive Swagger UI documentation.

### GET /health
Health check endpoint.

## 🤖 Supported Embedding Models

| Model | Description |
|-------|-------------|
| `qwen` (default) | Qwen/Qwen3-Embedding-0.6B - Multilingual |
| `bge` | BAAI/bge-small-en - English optimized |
| `m3` | BAAI/bge-m3 - Multilingual |
| `multilingual` | intfloat/multilingual-e5-base |

## 🧪 Testing

Run the test suite:

```bash
# Make sure API is running first
python test_api.py

# Or with a custom PDF
python test_api.py /path/to/document.pdf
```

Run the Python example:

```bash
python example_usage.py
```

## ⚙️ Configuration

Configure Qdrant connection via environment variables:

```bash
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=your-api-key  # Optional
```

## 📦 Dependencies

- **LlamaIndex**: Document processing and RAG framework
- **FastAPI**: REST API framework
- **Qdrant**: Vector database
- **HuggingFace Transformers**: Embedding models
- **PyMuPDF**: PDF processing

## 🔒 Security

All code has been scanned with CodeQL. No security vulnerabilities found.

## 📝 License

See the repository license for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please open an issue on GitHub.
