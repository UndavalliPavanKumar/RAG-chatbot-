"""FastAPI backend for the RAG Chatbot."""

import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pdf_loader import PDFLoader
from backend.vector_store import VectorStore
from backend.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Initialize components
pdf_loader = PDFLoader(data_dir=os.getenv("DATA_DIR", "./data"))
vector_store = VectorStore(store_path=os.getenv("VECTOR_STORE_PATH", "./faiss_db"))
rag_pipeline = RAGPipeline(vector_store)

app = FastAPI(title="RAG Chatbot API")


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    question: str


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    sources: List[str] = []


@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup."""
    print("Starting RAG Chatbot API...")
    vector_store.load()
    
    if vector_store.faiss_index is None or vector_store.faiss_index.ntotal == 0:
        print("No existing vector store found. Loading documents...")
        try:
            documents = pdf_loader.load_pdfs()
            
            if documents:
                vector_store.create_from_documents(documents)
            else:
                print("No PDF documents found in data directory")
        except Exception as e:
            print(f"Warning: Could not load documents during startup: {str(e)}")
            print("You can manually reload documents via the /reload-documents endpoint after setting OPENAI_API_KEY")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot API is running",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using the RAG pipeline.
    
    Args:
        request: QueryRequest with the question
        
    Returns:
        QueryResponse with answer and sources
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = rag_pipeline.query(request.question)
        
        sources = []
        for doc in result.get("source_documents", []):
            source_info = f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})"
            sources.append(source_info)
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/reload-documents")
async def reload_documents():
    """Reload documents from the data directory."""
    try:
        documents = pdf_loader.load_pdfs()
        
        if not documents:
            return {"status": "error", "message": "No PDF documents found in data directory"}
        
        vector_store.create_from_documents(documents)
        
        return {
            "status": "success",
            "message": f"Loaded {len(documents)} documents",
            "documents_count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading documents: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store.faiss_index is not None and vector_store.faiss_index.ntotal > 0
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)
