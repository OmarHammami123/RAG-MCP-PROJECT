"""
Centralized logging configuration for RAG MCP Project

Features:
- Rotating file logs (10MB max, 5 backups)
- Console + file output
- Structured logging with timestamps
- Different log levels per module
- Query performance tracking
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json


class RAGLogger:
    """enhanced logger for rag system with performance tracking"""
    
    def __init__(self, name: str ="rag_system", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        #perfermance tracking
        self.query_stats={
            'total_queries': 0,
            'cache_hits': 0,
            'rag_success': 0,
            'web_fallbacks': 0,
            'errors': 0
        }
        self._setup_logger()
        
    def _setup_logger(self):
        """configure logging with file rotation and console output"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        #remove existing handlers to avaid duplicates
        self.logger.handlers.clear()
        
        #1 rotating file handler(main log)
        main_log = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            main_log,
            maxBytes=10*1024*1024, #10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        #2 console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        #3 error file handler(errors only)
        error_log = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=5*1024*1024, #5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        #add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)
        
    def log_query(self, query: str, source : str, duration : float, success: bool):
        """log query with performance details"""
        self.query_stats['total_queries'] += 1
        
        if source == 'cache':
            self.query_stats['cache_hits'] += 1
        elif source == 'rag':
            self.query_stats['rag_success'] += 1
        elif source == 'web':
            self.query_stats['web_fallbacks'] += 1       
        
        
        if not success:
            self.query_stats['errors'] += 1
        
        log_data= {
            'query': query,
            'source': source,
            'duration_ms': round(duration * 1000, 2),
            'success': success,
            }
        if success:
            self.logger.info(f"Query Processed: {json.dumps(log_data, ensure_ascii=False)}")             
        else:
            self.logger.error(f"Query Failed: {json.dumps(log_data, ensure_ascii=False)}")    
        
        
    def log_cache_operation(self, operation: str, key: str, hit: bool =False):
        """log cache operations"""
        log_data= {
            'operation': operation,
            "key": key[:50],
            'hit': hit
        }
        self.logger.debug(f"CACHE: {json.dumps(log_data, ensure_ascii=False)}")
        
    def log_mcp_tool(self, tool: str, params : Dict, success: bool):
        """log mcp tool usage"""
        log_data = {
            'tool': tool,
            'params': str(params)[:100],
            'success': success
        }
        
        if success:
            self.logger.info(f"MCP TOOL: {json.dumps(log_data, ensure_ascii=False)}")
        else:
            self.logger.error(f"MCP TOOL ERROR: {json.dumps(log_data, ensure_ascii=False)}")
        
    def log_document_processing(self, fileneame : str, chunks: int, duration : float):
        """log document processing details"""
        log_data = {
            'filename': fileneame,
            'chunks_created': chunks,
            'duration_s': round(duration, 2)
        }
        self.logger.info(f"DOC PROCESSING: {json.dumps(log_data, ensure_ascii=False)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """retrieve current query statistics"""
        return {
            **self.query_stats,
            'cache_hit_rate': round(
                (self.query_stats['cache_hits'] / self.query_stats['total_queries'] * 100) 
                if self.query_stats['total_queries'] > 0 else 0, 2)
        }    
       
    def export_stats(self, filepath: str = None):
        """export stats to json file"""
        if filepath is None:
            filepath = self.log_dir / f"{self.name}_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats= {
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats()
        }                    
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Exported stats to {filepath}")
        
        
    #convenience methods for logging at different levels
    def info(self, message: str):
        self.logger.info(message)
    def debug(self, message: str):
        self.logger.debug(message)
    def warning(self, message: str):
        self.logger.warning(message)
    def error(self, message: str, exc_info: bool =False):
        self.logger.error(message, exc_info=exc_info)
    def critical(self, message: str, exc_info: bool =False):
        self.logger.critical(message, exc_info=exc_info)
        
        
rag_logger = RAGLogger()

if __name__ == "__main__":
    print("Testing RAG Logger...\n")
    
    # Test different log levels
    rag_logger.info("System initialized")
    rag_logger.debug("Debug information")
    rag_logger.warning("This is a warning")
    rag_logger.error("This is an error")
    
    # Test query logging
    rag_logger.log_query("What is LEAN?", "cache", 0.001, success=True)
    rag_logger.log_query("Another question", "rag", 2.5, success=True)
    rag_logger.log_query("Failed query", "rag", 1.2, success=False)
    
    # Test MCP tool logging
    rag_logger.log_mcp_tool("list_files", {"path": "."}, success=True)
    
    # Test document processing
    rag_logger.log_document_processing("test.pdf", 150, 5.2)
    
    # Print stats
    print("\n" + "="*60)
    print("Logger Statistics:")
    print("="*60)
    stats = rag_logger.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export stats
    rag_logger.export_stats()
    
    print("\n Check the 'logs/' directory for output files!")
                