"""
Document upload module for loading and indexing documents into Qdrant vector store.
"""
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_connection import vector_store, index
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
import os

# Default embedding model
embed_model = None


def set_embed_model(model_name: str):
    """Set the embedding model to use."""
    global embed_model
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    return embed_model


def upload_document(file_path: str, embed_model_instance):
    """
    Upload and index a document into the vector store.
    
    Args:
        file_path: Path to the document file
        embed_model_instance: The embedding model to use for indexing
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False
    
    try:
        # Load documents
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index with custom embedding model
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model_instance,
            show_progress=True,
        )
        
        return True
    except Exception as e:
        print(f"Error uploading document: {e}")
        return False
