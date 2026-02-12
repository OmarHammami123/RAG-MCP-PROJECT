import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict
import threading
from exceptions import CacheLoadError, CacheSaveError, handle_exception
from logging_config import rag_logger

class QueryCache:
    """
    query caching system with ttl (time to live) and persistence
    features:
    - in memory cache for fast access
    - disk persistence for cache survival across restarts
    - ttl based expiration 
    - thread-safe operations
    - cache statistics tracking
    """
    
    def __init__(self, ttl_minutes: int = 60, cache_dir: str= "cache",
                 persist: bool = True, max_entries: int = 1000):
        """
        initialize query cache
        args:
            ttl_minutes: time to live for cache entries in minutes
            cache_dir: directory to store cache files
            persist: whether to persist cache to disk
            max_entries: maximum number of entries in the cache before cleanup
        """
        self.cache: Dict[str, tuple] = {} #{key: (result ,timestamp, hits)}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.persist = persist
        self.max_entries = max_entries
        self.lock = threading.Lock() #thread safety
        self.logger = rag_logger
        
        #stats
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_queries': 0
        }
        
        #setup persistence 
        if self.persist:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir( exist_ok=True)
            self.cache_file = self.cache_dir / "query_cache.pkl"
            self._load_from_disk()
    
    
    def get_cache_key(self, query: str , k: int = 5, **kwargs) ->str:
        """
        generate cache key from query and params
        args:
            query: user question
            k: nb of sources
            **kwargs: additional parameters affecting results
            
        returns:
        md5 hash of normalized query and params    
        """
        #normalize query (lowercase , strip whitespace)
        normalized_query = query.lower().strip()
        #include params that affect results
        cache_params = {
            'query': normalized_query,
            'k': k,
            **kwargs    
        }
        #create deterministic string representation
        param_string = json.dumps(cache_params, sort_keys=True)
        
        #generate md5 hash
        return hashlib.md5(param_string.encode()).hexdigest()
    
    def get(self, query: str, k: int =5, **kwargs) -> Optional[Dict[str, Any]]:
        """
        retrieve cached result 
        args:
            query: user question
            k: nb of sources
            **kwargs: additional parameters affecting results
            returns:
            cached result or None if not found/expired
            """
        with self.lock:
            self.stats['total_queries'] += 1
            key  = self.get_cache_key(query, k, **kwargs)
            
            if key in self.cache:
                result, timestamp, hits = self.cache[key]
                #check if expired
                if datetime.now() - timestamp < self.ttl:
                    #update hit count
                    self.cache[key]= (result, timestamp, hits + 1)
                    self.stats['hits'] += 1
                    
                    print(f"[CACHE HIT] Query: '{query[:50]}...' (hit #{hits + 1})")
                    return result
                else:
                    #expired remove it
                    del self.cache[key]
                    print(f"[CACHE EXPIRED] Query: '{query[:50]}...'")
            self.stats['misses'] += 1
            print(f"[CACHE MISS] Query: '{query[:50]}...'")
            return None
        
    
    
    def set(self, query: str , result: Dict[str, Any], k: int =5, **kwargs):
        """
        store result in cache
        args:
            query: user question
            result: result to cache
            k: nb of sources
            **kwargs: additional parameters affecting results
        """
        with self.lock:
            key = self.get_cache_key(query, k, **kwargs)
            self.cache[key] = (result, datetime.now(), 0)
            print(f"[CACHE STORE] Query: '{query[:50]}...' stored in cache.")
            
            #check if cleanup needed
            if len(self.cache) > self.max_entries:
                self._evict_old_entries()
            
            #persist to disk
            if self.persist:
                self._save_to_disk()
                
                
    def _cleanup_old_entries(self):
        """remove expired entries from cache"""
        now = datetime.now()
        expired_keys = [
            k for k , (_,timestamp, _) in self.cache.items()
            if now - timestamp >= self.ttl
        ]            
        for k in expired_keys:
            del self.cache[k]
            self.stats['evictions'] += 1
        print (f"[CACHE CLEANUP] Removed {len(expired_keys)} expired entries.")
        #if still too many entries, evict oldest
        if len(self.cache) > self.max_entries:
            #sort by timestamp
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda item: item[1][2] #hits count
                
            )
            #remove bottom 20%
            remove_count = int(0.2*len(sorted_entries))
            for key, _ in sorted_entries[:remove_count]:
                del self.cache[key]
                self.stats['evictions'] += 1
            print(f"[CACHE EVICTION] Evicted {remove_count} least used entries.")
    
    
    def clear(self):
        """clear entire cache """
        with self.lock:
            self.cache.clear()
            print("[CACHE CLEAR] Cache cleared.")
            
            if self.persist and self.cache_file.exists():
                self.cache_file.unlink()
                print("[CACHE CLEAR] Cache file deleted from disk.")
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats['hits'] / self.stats['total_queries'] * 100 
                       if self.stats['total_queries'] > 0 else 0)
            
            return {
                'total_queries': self.stats['total_queries'],
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': f"{hit_rate:.2f}%",
                'cached_entries': len(self.cache),
                'evictions': self.stats['evictions'],
                'ttl_minutes': self.ttl.total_seconds() / 60
            }
    
    def _save_to_disk(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'stats': self.stats
                }, f)
            self.logger.debug(f"Cache saved to {self.cache_file}")
        except Exception as e:
            cache_error = CacheSaveError(
                str(self.cache_file),
                original_error=str(e)
            )
            handle_exception(cache_error, logger=self.logger)
    
    def _load_from_disk(self):
        """Load cache from disk"""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.cache = data.get('cache', {})
                self.stats = data.get('stats', self.stats)
            
            # Clean expired entries on load
            self._cleanup_old_entries()
            
            self.logger.info(f"Loaded {len(self.cache)} entries from cache")
        except Exception as e:
            cache_error = CacheLoadError(
                str(self.cache_file),
                original_error=str(e)
            )
            handle_exception(cache_error, logger=self.logger)
            self.cache = {}                         
                