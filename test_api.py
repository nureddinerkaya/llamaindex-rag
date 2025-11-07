"""
Test script for the RAG API.
This script demonstrates how to use the API endpoints.

Note: Requires all dependencies to be installed and Qdrant to be running.
"""
import requests
import json


BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_root():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_upload(file_path, embed_model="qwen"):
    """Test the file upload endpoint."""
    print(f"Testing upload endpoint with file: {file_path}, model: {embed_model}")
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path.split("/")[-1], f, "application/pdf")}
        data = {"embed_model": embed_model}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    return response.status_code == 200


def test_query(query_text, embed_model="qwen", top_k=3):
    """Test the query endpoint."""
    print(f"Testing query endpoint...")
    print(f"Query: {query_text}")
    print(f"Model: {embed_model}, Top-K: {top_k}")
    
    payload = {
        "query": query_text,
        "embed_model": embed_model,
        "top_k": top_k
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Results: {result['total_results']}")
        print("\nResults:")
        for i, res in enumerate(result['results'], 1):
            print(f"\n{i}. Score: {res['score']}")
            print(f"   Content: {res['content'][:200]}...")
            print(f"   Metadata: {res['metadata']}")
    else:
        print(f"Error: {response.json()}")
    
    print()


def main():
    """Run all tests."""
    import sys
    import os
    
    print("=" * 70)
    print("RAG API Test Suite")
    print("=" * 70)
    print()
    
    try:
        # Test health and root endpoints
        test_health_check()
        test_root()
        
        # Test file upload
        # Check for PDF file path from command line or use default
        if len(sys.argv) > 1:
            pdf_path = sys.argv[1]
        else:
            pdf_path = "./data/llama2.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            print("Usage: python test_api.py [path_to_pdf]")
            return
        
        upload_success = test_upload(pdf_path, embed_model="qwen")
        
        if upload_success:
            print("Upload successful! Proceeding with query test...")
            print()
            
            # Test query
            test_query(
                query_text="What is fine-tuning?",
                embed_model="qwen",
                top_k=3
            )
            
            # Test with different model
            test_query(
                query_text="Explain transformers",
                embed_model="bge",
                top_k=5
            )
        else:
            print("Upload failed. Skipping query tests.")
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the API server is running:")
        print("  python api.py")
        print("or")
        print("  uvicorn api:app --host 0.0.0.0 --port 8000")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the PDF file exists at the specified path.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print()
    print("=" * 70)
    print("Tests completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
