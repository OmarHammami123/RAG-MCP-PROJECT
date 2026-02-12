"""
Custom exceptions for RAG MCP Project

Hierarchy:
- RAGException (base)
  ├── VectorStoreException
  │   ├── VectorStoreInitError
  │   ├── VectorStoreQueryError
  │   └── EmbeddingError
  ├── DocumentProcessingException
  │   ├── DocumentLoadError
  │   ├── DocumentParseError
  │   └── ChunkingError
  ├── MCPException
  │   ├── MCPConnectionError
  │   ├── MCPToolError
  │   └── MCPSecurityError
  ├── CacheException
  │   ├── CacheLoadError
  │   └── CacheSaveError
  └── LLMException
      ├── LLMConnectionError
      ├── LLMQuotaError
      └── LLMResponseError
"""


class RAGException(Exception):
    """Base exception for all RAG-related errors"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============ Vector Store Exceptions ============

class VectorStoreException(RAGException):
    """Base exception for vector store operations"""
    pass


class VectorStoreInitError(VectorStoreException):
    """Raised when vector store initialization fails"""
    def __init__(self, message: str = "Failed to initialize vector store", **kwargs):
        super().__init__(message, kwargs)


class VectorStoreQueryError(VectorStoreException):
    """Raised when vector store query fails"""
    def __init__(self, query: str, message: str = "Vector store query failed", **kwargs):
        super().__init__(message, {'query': query, **kwargs})


class EmbeddingError(VectorStoreException):
    """Raised when embedding generation fails"""
    def __init__(self, text: str, message: str = "Embedding generation failed", **kwargs):
        super().__init__(message, {'text_preview': text[:100], **kwargs})


# ============ Document Processing Exceptions ============

class DocumentProcessingException(RAGException):
    """Base exception for document processing"""
    pass


class DocumentLoadError(DocumentProcessingException):
    """Raised when document loading fails"""
    def __init__(self, filepath: str, message: str = "Failed to load document", **kwargs):
        super().__init__(message, {'filepath': filepath, **kwargs})


class DocumentParseError(DocumentProcessingException):
    """Raised when document parsing fails"""
    def __init__(self, filepath: str, page: int = None, message: str = "Failed to parse document", **kwargs):
        details = {'filepath': filepath, **kwargs}
        if page:
            details['page'] = page
        super().__init__(message, details)


class ChunkingError(DocumentProcessingException):
    """Raised when document chunking fails"""
    def __init__(self, document: str, message: str = "Failed to chunk document", **kwargs):
        super().__init__(message, {'document': document, **kwargs})


# ============ MCP Exceptions ============

class MCPException(RAGException):
    """Base exception for MCP operations"""
    pass


class MCPConnectionError(MCPException):
    """Raised when MCP server connection fails"""
    def __init__(self, message: str = "Failed to connect to MCP server", **kwargs):
        super().__init__(message, kwargs)


class MCPToolError(MCPException):
    """Raised when MCP tool execution fails"""
    def __init__(self, tool: str, params: dict, message: str = "MCP tool execution failed", **kwargs):
        super().__init__(message, {'tool': tool, 'params': params, **kwargs})


class MCPSecurityError(MCPException):
    """Raised when MCP security validation fails (e.g., path traversal)"""
    def __init__(self, path: str, message: str = "Security validation failed", **kwargs):
        super().__init__(message, {'attempted_path': path, **kwargs})


# ============ Cache Exceptions ============

class CacheException(RAGException):
    """Base exception for cache operations"""
    pass


class CacheLoadError(CacheException):
    """Raised when cache loading from disk fails"""
    def __init__(self, filepath: str, message: str = "Failed to load cache", **kwargs):
        super().__init__(message, {'filepath': filepath, **kwargs})


class CacheSaveError(CacheException):
    """Raised when cache saving to disk fails"""
    def __init__(self, filepath: str, message: str = "Failed to save cache", **kwargs):
        super().__init__(message, {'filepath': filepath, **kwargs})


# ============ LLM Exceptions ============

class LLMException(RAGException):
    """Base exception for LLM operations"""
    pass


class LLMConnectionError(LLMException):
    """Raised when LLM connection fails"""
    def __init__(self, model: str, message: str = "Failed to connect to LLM", **kwargs):
        super().__init__(message, {'model': model, **kwargs})


class LLMQuotaError(LLMException):
    """Raised when LLM API quota is exceeded"""
    def __init__(self, model: str, message: str = "API quota exceeded", **kwargs):
        super().__init__(message, {'model': model, **kwargs})


class LLMResponseError(LLMException):
    """Raised when LLM returns invalid/unexpected response"""
    def __init__(self, response: str, message: str = "Invalid LLM response", **kwargs):
        super().__init__(message, {'response_preview': response[:200], **kwargs})


# ============ Utility Functions ============

def handle_exception(exception: Exception, logger=None, reraise: bool = False):
    """
    Centralized exception handler
    
    Args:
        exception: The caught exception
        logger: Optional logger instance
        reraise: Whether to re-raise after handling
    """
    if logger:
        if isinstance(exception, RAGException):
            logger.error(f"{exception.__class__.__name__}: {exception}", exc_info=True)
        else:
            logger.error(f"Unexpected error: {exception}", exc_info=True)
    else:
        print(f"[ERROR] {exception.__class__.__name__}: {exception}")
    
    if reraise:
        raise


# Test script
if __name__ == "__main__":
    print("Testing Custom Exceptions...\n")
    
    # Test VectorStoreException
    try:
        raise VectorStoreQueryError("test query", reason="timeout")
    except VectorStoreQueryError as e:
        print(f"✓ VectorStoreQueryError: {e}")
    
    # Test DocumentProcessingException
    try:
        raise DocumentLoadError("/path/to/file.pdf", error_code=404)
    except DocumentLoadError as e:
        print(f"✓ DocumentLoadError: {e}")
    
    # Test MCPException
    try:
        raise MCPSecurityError("../../etc/passwd", reason="path_traversal")
    except MCPSecurityError as e:
        print(f"✓ MCPSecurityError: {e}")
    
    # Test LLMException
    try:
        raise LLMQuotaError("gemini-2.0-flash", limit=60, used=61)
    except LLMQuotaError as e:
        print(f"✓ LLMQuotaError: {e}")
    
    print("\n All exception types working correctly!")