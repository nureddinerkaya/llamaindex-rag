#!/usr/bin/env python3
"""
CLI Application for RAG (Retrieval-Augmented Generation) using LlamaIndex and Qdrant.

This CLI allows users to:
1. Select an embedding model
2. Upload documents to the vector store
3. Query the vector store and retrieve relevant documents
"""

import os
from yukleme import set_embed_model, upload_document
from qdrant_connection import vector_store
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Optional


# Available embedding models
AVAILABLE_MODELS = {
    "1": {
        "name": "bge-m3",
        "model_path": "BAAI/bge-m3"
    },
    "2": {
        "name": "multilingual-intfloat",
        "model_path": "intfloat/multilingual-e5-large"
    }
}

# Content preview length for query results
CONTENT_PREVIEW_LENGTH = 500


def display_model_menu():
    """Display available embedding models."""
    print("\n=== Embedding Model Selection ===")
    for key, model_info in AVAILABLE_MODELS.items():
        print(f"{key}. {model_info['name']} ({model_info['model_path']})")
    print("=" * 35)


def select_model():
    """Prompt user to select an embedding model."""
    display_model_menu()
    
    while True:
        choice = input("\nSelect embedding model (1-2): ").strip()
        if choice in AVAILABLE_MODELS:
            selected_model = AVAILABLE_MODELS[choice]
            print(f"\n✓ Selected model: {selected_model['name']}")
            return selected_model['model_path']
        else:
            print("Invalid choice. Please select 1 or 2.")


def display_action_menu():
    """Display available actions."""
    print("\n=== Action Selection ===")
    print("1. Upload document")
    print("2. Query documents")
    print("=" * 25)


def select_action():
    """Prompt user to select an action."""
    display_action_menu()
    
    while True:
        choice = input("\nSelect action (1-2): ").strip()
        if choice in ["1", "2"]:
            return choice
        else:
            print("Invalid choice. Please select 1 or 2.")


def handle_file_upload(embed_model_instance):
    """Handle document upload."""
    print("\n=== Document Upload ===")
    filename = input("Enter filename (file should be in the same directory): ").strip()
    
    # Construct full path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    
    # Also check if file exists in data directory
    if not os.path.exists(file_path):
        data_file_path = os.path.join(current_dir, "data", filename)
        if os.path.exists(data_file_path):
            file_path = data_file_path
    
    print(f"\nUploading document: {filename}")
    print("Please wait, this may take a moment...")
    
    success = upload_document(file_path, embed_model_instance)
    
    if success:
        print("\n✓ SUCCESS: Document uploaded and indexed successfully!")
    else:
        print("\n✗ FAILED: Document upload failed.")


def handle_query(embed_model_instance):
    """Handle document query."""
    print("\n=== Document Query ===")
    query_str = input("Enter your query: ").strip()
    
    if not query_str:
        print("Query cannot be empty.")
        return
    
    # Get top-k parameter
    while True:
        try:
            top_k = input("Enter number of results to retrieve (default: 3): ").strip()
            top_k = int(top_k) if top_k else 3
            if top_k > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nSearching for: '{query_str}'")
    print("Please wait...\n")
    
    try:
        # Create query embedding
        query_embedding = embed_model_instance.get_query_embedding(query_str)
        
        # Query vector store
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            mode="default"
        )
        
        query_result = vector_store.query(vector_store_query)
        
        # Display results
        print(f"\n=== Top {top_k} Results ===\n")
        
        if not query_result.nodes:
            print("No results found.")
            return
        
        for index, node in enumerate(query_result.nodes, 1):
            score = None
            if query_result.similarities is not None and index - 1 < len(query_result.similarities):
                score = query_result.similarities[index - 1]
            
            print(f"Result {index}:")
            if score is not None:
                print(f"  Similarity Score: {score:.4f}")
            print(f"  Content:\n  {node.get_content()[:CONTENT_PREVIEW_LENGTH]}...")
            print("-" * 80)
            
    except Exception as e:
        print(f"\n✗ Error during query: {e}")


def main():
    """Main CLI application loop."""
    print("\n" + "=" * 50)
    print(" RAG CLI Application - LlamaIndex + Qdrant")
    print("=" * 50)
    
    # Step 1: Select embedding model
    model_path = select_model()
    embed_model_instance = set_embed_model(model_path)
    
    # Step 2: Select action
    action = select_action()
    
    # Step 3: Execute action
    if action == "1":
        handle_file_upload(embed_model_instance)
    elif action == "2":
        handle_query(embed_model_instance)
    
    print("\n" + "=" * 50)
    print(" Thank you for using RAG CLI!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
