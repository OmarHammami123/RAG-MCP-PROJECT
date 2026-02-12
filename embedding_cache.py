# filepath: embedding_cache.py
"""
Embedding Cache System for RAG MCP Project

Purpose: Speed up document processing by caching embeddings
- Reduces re-computation of embeddings for unchanged documents
- Uses file hash to detect changes
- Persistent disk storage
- Automatic cache invalidation
"""

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from exceptions import CacheLoadError, CacheSaveError, handle_exception
from logging_config import rag_logger


class EmbeddingCache:
    """Cache for document embeddings to speed up processing"""
    
    def __init__(self, cache_dir: str = "cache/embeddings"):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = rag_logger
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }
        self.logger.info(f"Embedding cache initialized at {self.cache_dir}")
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of file content
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to hash file {file_path}: {e}")
            return None
    
    def get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for a given file hash"""
        return self.cache_dir / f"{file_hash}.pkl"
    
    def get(self, file_path: str) -> Optional[Dict]:
        """
        Retrieve cached embeddings for a file
        
        Args:
            file_path: Path to original file
            
        Returns:
            Cached data dict with 'embeddings', 'metadata', 'timestamp' or None
        """
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            self.stats['misses'] += 1
            return None
        
        cache_path = self.get_cache_path(file_hash)
        
        if not cache_path.exists():
            self.logger.debug(f"Cache miss for {Path(file_path).name}")
            self.stats['misses'] += 1
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.logger.debug(f"Cache hit for {Path(file_path).name}")
            self.stats['hits'] += 1
            return cached_data
            
        except Exception as e:
            cache_error = CacheLoadError(
                str(cache_path),
                original_error=str(e)
            )
            handle_exception(cache_error, logger=self.logger)
            self.stats['misses'] += 1
            return None
    
    def set(self, file_path: str, embeddings: List, metadata: Dict = None):
        """
        Cache embeddings for a file
        
        Args:
            file_path: Path to original file
            embeddings: List of embedding vectors
            metadata: Optional metadata about the embeddings
        """
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_path = self.get_cache_path(file_hash)
        
        cache_data = {
            'embeddings': embeddings,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path),
            'file_hash': file_hash
        }
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.debug(f"Cached embeddings for {Path(file_path).name}")
            self.stats['saves'] += 1
            return True
            
        except Exception as e:
            cache_error = CacheSaveError(
                str(cache_path),
                original_error=str(e)
            )
            handle_exception(cache_error, logger=self.logger)
            return False
    
    def invalidate(self, file_path: str):
        """
        Remove cached embeddings for a file
        
        Args:
            file_path: Path to original file
        """
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return False
        
        cache_path = self.get_cache_path(file_hash)
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                self.logger.info(f"Invalidated cache for {Path(file_path).name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to invalidate cache: {e}")
                return False
        return False
    
    def clear_all(self):
        """Clear all cached embeddings"""
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1
            
            self.logger.info(f"Cleared {count} cached embeddings")
            self.stats = {'hits': 0, 'misses': 0, 'saves': 0}
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_queries = self.stats['hits'] + self.stats['misses']
        hit_rate = (
            f"{(self.stats['hits'] / total_queries * 100):.2f}%"
            if total_queries > 0 else "0%"
        )
        
        # Count cached files
        cached_files = len(list(self.cache_dir.glob("*.pkl")))
        
        # Calculate total cache size
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        size_mb = total_size / (1024 * 1024)
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'saves': self.stats['saves'],
            'hit_rate': hit_rate,
            'cached_files': cached_files,
            'cache_size_mb': round(size_mb, 2)
        }
    
    def list_cached_files(self) -> List[Dict]:
        """List all cached files with metadata"""
        cached = []
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                cached.append({
                    'file_path': data.get('file_path', 'Unknown'),
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'file_hash': data.get('file_hash', 'Unknown'),
                    'cache_file': str(cache_file)
                })
            except Exception as e:
                self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
        
        return cached


# Test script
if __name__ == "__main__":
    print("Testing Embedding Cache System...\n")
    
    cache = EmbeddingCache()
    
    # Simulate some embeddings
    fake_embeddings = [[0.1, 0.2, 0.3] for _ in range(10)]
    test_file = "test_document.pdf"
    
    # Test 1: Cache miss
    print("1. Testing cache miss...")
    result = cache.get(test_file)
    assert result is None, "Should return None on cache miss"
    print("   ✓ Cache miss works\n")
    
    # Test 2: Cache save
    print("2. Testing cache save...")
    success = cache.set(test_file, fake_embeddings, {'chunks': 10})
    assert success is False, "File doesn't exist, should fail"
    print("   ✓ Cache save validation works\n")
    
    # Test 3: Stats
    print("3. Testing cache statistics...")
    stats = cache.get_stats()
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']}")
    print(f"   Cached Files: {stats['cached_files']}")
    print(f"   Cache Size: {stats['cache_size_mb']} MB\n")
    
    print("✓ All embedding cache tests passed!")
