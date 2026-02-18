"""
Script to start the FastAPI server
Usage: uv run start_api.py
"""

import uvicorn
from logging_config import rag_logger

if __name__ == "__main__":
    rag_logger.info("Starting RAG System API Server...")
    rag_logger.info("API Documentation: http://localhost:8000/docs")
    rag_logger.info("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
