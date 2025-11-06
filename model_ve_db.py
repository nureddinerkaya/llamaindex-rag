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
embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B")


# When running Qdrant locally in Docker, the default URL is http://localhost:6333
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Create a qdrant client. For local instances, set QDRANT_URL to your local host
client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Expose vector_store and index (used by yukleme.py)
vector_store = QdrantVectorStore(client=client, collection_name="faq")
# Pass the explicitly configured embed_model to avoid resolving the default (OpenAI) embedder on import
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
