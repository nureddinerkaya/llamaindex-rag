# qdrant_connection.py
"""
Compatibility shim for older file named `qdrant-connection.py`.
Expose `vector_store` and `index` for imports in `yukleme.py`.
Reads Qdrant connection info from environment variables if available.
"""
import os

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# 0. Embed Model
try:
    import torch
except Exception:
    torch = None

# Detect CUDA and set model kwargs to prefer GPU (fp16) when available
cuda_available = (torch is not None) and torch.cuda.is_available()

if cuda_available:
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": getattr(torch, "float16", None),
        "low_cpu_mem_usage": True,
    }
else:
    model_kwargs = {"device_map": "cpu"}

embed_model = HuggingFaceEmbedding(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs=model_kwargs,
)

# When running Qdrant locally in Docker, the default URL is http://localhost:6333
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Create a qdrant client. For local instances, set QDRANT_URL to your local host
client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Expose vector_store and index (used by yukleme.py)
vector_store = QdrantVectorStore(client=client, collection_name="kotku")
# Pass the explicitly configured embed_model to avoid resolving the default (OpenAI) embedder on import
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
