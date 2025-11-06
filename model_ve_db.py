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


def get_embed_model(model_name: str = "qwen"):
    """Get embedding model by name.
    
    Args:
        model_name: Name of the model to use. Options:
            - "qwen" (default): Qwen/Qwen3-Embedding-0.6B
            - "bge": BAAI/bge-small-en
            - "m3": BAAI/bge-m3
            - "multilingual": intfloat/multilingual-e5-base
    
    Returns:
        HuggingFaceEmbedding instance
    """
    model_map = {
        "qwen": "Qwen/Qwen3-Embedding-0.6B",
        "bge": "BAAI/bge-small-en",
        "m3": "BAAI/bge-m3",
        "multilingual": "intfloat/multilingual-e5-base",
    }
    
    model_name_lower = model_name.lower()
    actual_model_name = model_map.get(model_name_lower, model_map["qwen"])
    
    return HuggingFaceEmbedding(model_name=actual_model_name)


# When running Qdrant locally in Docker, the default URL is http://localhost:6333
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Create a qdrant client. For local instances, set QDRANT_URL to your local host
client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Expose vector_store and index (used by yukleme.py)
vector_store = QdrantVectorStore(client=client, collection_name="documents")
# Pass the explicitly configured embed_model to avoid resolving the default (OpenAI) embedder on import
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
