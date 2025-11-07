"""
FastAPI application for RAG (Retrieval-Augmented Generation) system.

Provides endpoints for:
- File upload with optional embed model selection
- Query with optional embed model selection and top_k parameter
"""
import os
import tempfile
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field

from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore

from model_ve_db import vector_store, get_embed_model


app = FastAPI(
    title="RAG API",
    description="API for file upload and querying with RAG system",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="Query text to search for")
    embed_model: Optional[str] = Field("qwen", description="Embedding model to use (qwen, bge, m3, multilingual)")
    top_k: Optional[int] = Field(3, description="Number of top results to return", ge=1, le=100)


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    results: List[dict] = Field(..., description="List of query results with score and content")
    total_results: int = Field(..., description="Total number of results returned")


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Name of the uploaded file")
    nodes_count: int = Field(..., description="Number of nodes created and uploaded")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API",
        "endpoints": {
            "/upload": "POST - Upload a file with optional embed model selection",
            "/query": "POST - Query the vector store with optional parameters",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(..., description="File to upload (PDF supported)"),
    embed_model: str = Form("qwen", description="Embedding model to use (qwen, bge, m3, multilingual)")
):
    """
    Upload a file and index it in the vector store.
    
    Args:
        file: File to upload (PDF format)
        embed_model: Embedding model to use (default: qwen)
    
    Returns:
        UploadResponse with status message and number of nodes created
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Get the embedding model
    embedding_model = get_embed_model(embed_model)
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load and process the file
        loader = PyMuPDFReader()
        documents = loader.load(file_path=tmp_file_path)
        
        # Split documents into chunks
        text_parser = SentenceSplitter(chunk_size=1024)
        
        text_chunks = []
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))
        
        # Create nodes from text chunks
        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)
        
        # Generate embeddings for each node
        # Extract all text content for batch embedding
        texts_to_embed = [node.get_content() for node in nodes]
        
        # Get embeddings in batch (if the model supports it) or one by one
        try:
            # Try batch embedding first (more efficient)
            embeddings = embedding_model.get_text_embedding_batch(texts_to_embed)
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
        except (AttributeError, NotImplementedError):
            # Fall back to individual embeddings if batch is not supported
            for node in nodes:
                node_embedding = embedding_model.get_text_embedding(node.get_content())
                node.embedding = node_embedding
        
        # Add nodes to vector store
        vector_store.add(nodes)
        
        return UploadResponse(
            message="File uploaded and indexed successfully",
            filename=file.filename,
            nodes_count=len(nodes)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the vector store for similar documents.
    
    Args:
        request: QueryRequest with query text, optional embed_model and top_k
    
    Returns:
        QueryResponse with list of results and scores
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query text cannot be empty"
        )
    
    # Get the embedding model
    embedding_model = get_embed_model(request.embed_model)
    
    try:
        # Create query embedding
        query_embedding = embedding_model.get_query_embedding(request.query)
        
        # Query vector store
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=request.top_k,
            mode="default"
        )
        
        query_result = vector_store.query(vector_store_query)
        
        # Parse results into nodes with scores
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
        # Format response
        results = []
        for nws in nodes_with_scores:
            results.append({
                "score": nws.score,
                "content": nws.node.get_content(),
                "metadata": nws.node.metadata
            })
        
        return QueryResponse(
            results=results,
            total_results=len(results)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying documents: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
