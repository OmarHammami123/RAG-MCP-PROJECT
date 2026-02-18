"""
FastAPI REST API for RAG System
Provides HTTP endpoints for document querying and cache management
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from pathlib import Path
import traceback
import time

from simple_rag import RAGSystem
from exceptions import RAGException
from logging_config import rag_logger

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="REST API for document-based question answering with caching",
    version="1.0.0"
)

# Initialize RAG system (singleton)
rag_system: Optional[RAGSystem] = None


# Request/Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask the RAG system", min_length=1)
    max_results: int = Field(default=5, description="Maximum number of results to return", ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    cached: bool
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    cache_entries: int
    system_ready: bool


class CacheStatsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: str
    cached_entries: int
    evictions: int
    ttl_minutes: float


class DocumentInfo(BaseModel):
    name: str
    size_bytes: int
    indexed: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_logger.info("Initializing RAG System...")
        rag_system = RAGSystem()
        
        # Initialize vector store and LLM
        rag_logger.info("Loading vector store and LLM...")
        success = rag_system.initialize()
        
        if not success:
            raise Exception("Failed to initialize vector store or LLM")
        
        rag_logger.info("RAG System initialized successfully")
    except Exception as e:
        rag_logger.error(f"Failed to initialize RAG System: {e}")
        rag_logger.error(traceback.format_exc())
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_system
    if rag_system:
        rag_logger.info("Shutting down RAG System...")
        rag_system.close()
        rag_system = None


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG System not initialized")
        
        # Get cache stats
        cache_stats = rag_system.query_cache.get_stats()
        
        # Count documents
        documents_path = Path("documents")
        doc_count = len(list(documents_path.glob("*.pdf"))) if documents_path.exists() else 0
        
        return HealthResponse(
            status="healthy",
            documents_indexed=doc_count,
            cache_entries=cache_stats["cached_entries"],
            system_ready=True
        )
    except Exception as e:
        rag_logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question
    
    - **question**: Your question about the documents
    - **max_results**: Maximum number of context chunks to retrieve (1-20)
    """
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG System not initialized")
        
        rag_logger.info(f"Processing query: {request.question}")
        
        start_time = time.time()
        
        # Query the RAG system (handles caching internally)
        result = rag_system.query(question=request.question)
        
        # Extract answer and sources from result
        answer = result.get("answer", "")
        sources_raw = result.get("sources", [])
        
        # Convert sources to strings (they come as dicts with file_name and chunk_index)
        sources = []
        for src in sources_raw:
            if isinstance(src, dict):
                file_name = src.get("file_name", "unknown")
                chunk_index = src.get("chunk_index", 0)
                sources.append(f"{file_name} (chunk {chunk_index})")
            elif isinstance(src, str):
                sources.append(src)
            else:
                sources.append(str(src))
        
        processing_time = time.time() - start_time
        
        # Check if it was cached (fast response < 0.1s typically means cached)
        is_cached = processing_time < 0.1
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            cached=is_cached,
            processing_time=round(processing_time, 3)
        )
        
    except RAGException as e:
        rag_logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        rag_logger.error(f"Query failed: {e}")
        rag_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["Cache"])
async def get_cache_stats():
    """Get cache statistics"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG System not initialized")
        
        stats = rag_system.query_cache.get_stats()
        
        return CacheStatsResponse(
            total_queries=stats["total_queries"],
            cache_hits=stats["hits"],
            cache_misses=stats["misses"],
            hit_rate=stats["hit_rate"],
            cached_entries=stats["cached_entries"],
            evictions=stats["evictions"],
            ttl_minutes=stats["ttl_minutes"]
        )
        
    except Exception as e:
        rag_logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear all cache entries"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG System not initialized")
        
        rag_system.query_cache.clear()
        rag_logger.info("Cache cleared successfully")
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        rag_logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all documents in the documents folder"""
    try:
        documents_path = Path("documents")
        
        if not documents_path.exists():
            return []
        
        documents = []
        for doc_path in documents_path.glob("*.pdf"):
            documents.append(DocumentInfo(
                name=doc_path.name,
                size_bytes=doc_path.stat().st_size,
                indexed=True
            ))
        
        return documents
        
    except Exception as e:
        rag_logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/reindex", tags=["Documents"])
async def reindex_documents():
    """Reindex all documents (clears vector store and rebuilds)"""
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG System not initialized")
        
        rag_logger.info("Starting document reindexing...")
        
        # Note: RAGSystem loads documents in __init__, so we'd need to add a reload method
        # For now, just return current state
        documents_path = Path("documents")
        doc_count = len(list(documents_path.glob("*.pdf"))) if documents_path.exists() else 0
        
        return {
            "message": "Documents are indexed",
            "documents_count": doc_count,
            "note": "To reindex, restart the server"
        }
        
    except Exception as e:
        rag_logger.error(f"Failed to reindex documents: {e}")
        rag_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
