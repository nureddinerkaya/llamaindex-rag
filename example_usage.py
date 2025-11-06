"""
Simple example showing how to use the RAG API from Python.

This example demonstrates:
1. Uploading a PDF file
2. Querying the vector store
3. Handling responses
"""
import requests
import json
from pathlib import Path


# API base URL
API_URL = "http://localhost:8000"


def upload_document(file_path: str, embed_model: str = "qwen") -> dict:
    """
    Upload a PDF document to the vector store.
    
    Args:
        file_path: Path to the PDF file
        embed_model: Embedding model to use (qwen, bge, m3, multilingual)
    
    Returns:
        Response dictionary with upload status
    """
    url = f"{API_URL}/upload"
    
    # Prepare the file
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/pdf')}
        data = {'embed_model': embed_model}
        
        # Make the request
        response = requests.post(url, files=files, data=data)
    
    # Check if successful
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Upload successful!")
        print(f"  File: {result['filename']}")
        print(f"  Nodes created: {result['nodes_count']}")
        return result
    else:
        print(f"✗ Upload failed: {response.json()}")
        return None


def query_documents(query: str, embed_model: str = "qwen", top_k: int = 3) -> dict:
    """
    Query the vector store for similar documents.
    
    Args:
        query: The question or search query
        embed_model: Embedding model to use
        top_k: Number of results to return
    
    Returns:
        Response dictionary with query results
    """
    url = f"{API_URL}/query"
    
    # Prepare the query
    payload = {
        "query": query,
        "embed_model": embed_model,
        "top_k": top_k
    }
    
    # Make the request
    response = requests.post(url, json=payload)
    
    # Check if successful
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Query successful!")
        print(f"  Found {result['total_results']} results\n")
        
        # Print results
        for i, res in enumerate(result['results'], 1):
            print(f"Result {i}:")
            print(f"  Score: {res['score']:.4f}")
            print(f"  Content: {res['content'][:200]}...")
            if res.get('metadata'):
                print(f"  Metadata: {res['metadata']}")
            print()
        
        return result
    else:
        print(f"✗ Query failed: {response.json()}")
        return None


def main():
    """Main example workflow."""
    print("=" * 70)
    print("RAG API Python Example")
    print("=" * 70)
    print()
    
    # Example 1: Upload a document
    print("1. Uploading a document...")
    print("-" * 70)
    
    pdf_path = "./data/llama2.pdf"  # Change this to your PDF path
    
    try:
        upload_result = upload_document(pdf_path, embed_model="qwen")
        print()
        
        if upload_result:
            # Example 2: Query the documents
            print("2. Querying the documents...")
            print("-" * 70)
            
            # Query 1
            print("\nQuery 1: What is fine-tuning?")
            query_documents(
                query="What is fine-tuning?",
                embed_model="qwen",
                top_k=3
            )
            
            # Query 2
            print("\nQuery 2: Explain transformers")
            query_documents(
                query="Explain transformers",
                embed_model="qwen",
                top_k=2
            )
            
            # Example 3: Using a different embedding model
            print("\n3. Using a different embedding model...")
            print("-" * 70)
            query_documents(
                query="What is LLaMA?",
                embed_model="bge",
                top_k=3
            )
        
    except FileNotFoundError:
        print(f"Error: File not found: {pdf_path}")
        print("Please update the pdf_path variable to point to a valid PDF file.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the API is running at", API_URL)
        print("\nStart the API with:")
        print("  python api.py")
    except Exception as e:
        print(f"Error: {e}")
    
    print("=" * 70)
    print("Example completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
