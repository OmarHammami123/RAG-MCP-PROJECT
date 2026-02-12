from typing import List, Optional
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import Config
from document_processor import DocumentProcessor
from exceptions import VectorStoreInitError, VectorStoreQueryError, EmbeddingError, handle_exception
from logging_config import rag_logger
import os 


class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embedding_model = None
        self.vector_store = None
        self.collection_name = "smart_notes_collection"
        self.logger = rag_logger
        
    def initialize(self, use_local_embeddings=True):
        """initilize the vector store and embedding model"""
        try:
            #get embedding model (local by default to avoid quota issues)
            self.embedding_model = self.config.get_embedding_model(use_local=use_local_embeddings)
            
            #init chromaDB
            self.vector_store = Chroma(
                collection_name = self.collection_name,
                embedding_function = self.embedding_model,
                persist_directory = self.config.CHROMA_DB_PATH
            )    
            self.logger.info(f"Vector store initialized: {self.collection_name}")
            return True
        except Exception as e:
            init_error = VectorStoreInitError(
                collection_name=self.collection_name,
                original_error=str(e)
            )
            handle_exception(init_error, logger=self.logger)
            return False
    
    def add_documents(self, documents: List[Document]):
        """add documents to the vector store"""
        if self.vector_store is None:
            error = VectorStoreInitError(message="Vector store is not initialized")
            handle_exception(error, logger=self.logger)
            return False
        
        try:
            self.logger.debug(f"Adding {len(documents)} documents to vector store...")
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            self.logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            embedding_error = EmbeddingError(
                text=documents[0].page_content[:100] if documents else "",
                count=len(documents),
                original_error=str(e)
            )
            handle_exception(embedding_error, logger=self.logger)
            return False
    
    def add_documents_with_embeddings(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents with pre-calculated embeddings to bypass embedding computation"""
        if self.vector_store is None:
            error = VectorStoreInitError(message="Vector store is not initialized")
            handle_exception(error, logger=self.logger)
            return False
        
        try:
            self.logger.debug(f"Adding {len(documents)} documents with pre-calculated embeddings...")
            
            # Access the underlying ChromaDB collection
            collection = self.vector_store._collection
            
            # Prepare data for ChromaDB
            ids = [f"doc_{i}_{hash(doc.page_content)}" for i, doc in enumerate(documents)]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to ChromaDB with pre-calculated embeddings
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(documents)} documents with cached embeddings")
            return True
            
        except Exception as e:
            embedding_error = EmbeddingError(
                text=documents[0].page_content[:100] if documents else "",
                count=len(documents),
                original_error=str(e)
            )
            handle_exception(embedding_error, logger=self.logger)
            return False
        
    def search(self,query: str, k: int =5) -> List[Document]:
        """perform similarity search in the vector store"""
        if self.vector_store is None:
            error = VectorStoreInitError(message="Vector store is not initialized")
            handle_exception(error, logger=self.logger)
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            self.logger.debug(f"Found {len(results)} documents for query: {query[:50]}")
            return results
        except Exception as e:
            query_error = VectorStoreQueryError(
                query=query,
                original_error=str(e)
            )
            handle_exception(query_error, logger=self.logger)
            return []
    
    
    def get_collection_info(self):
        """get info about the chroma collection"""
        if self.vector_store:
            try:
                collection = self.vector_store._collection
                count = collection.count()
                self.logger.debug(f"Collection '{self.collection_name}' has {count} items")
                return count
            except Exception as e:
                self.logger.error(f"Error getting collection info: {e}")
                return 0
        return 0        
    
    
#test vector store fuctionality

if __name__ == "__main__":
    print("[RELOAD] Setting up Smart Notes Vector Store...")
    
    processor = DocumentProcessor()
    documents = processor.process_documents()
    if documents:
        print(f"[OK] Processed {len(documents)} documents.")
        
        # Initialize vector store with local embeddings
        vector_store = VectorStore()
        
        print("\n[START] Initializing vector store with local embeddings...")
        success = vector_store.initialize(use_local_embeddings=True)
        
        if success:
            print("[ADD] Adding documents to vector store...")
            add_success = vector_store.add_documents(documents)
            
            if add_success:
                print("[SUCCESS] Vector store setup complete!")
                
                # Test search functionality
                print("\n[SEARCH] Testing search functionality...")
                test_query = "quality control methods"
                results = vector_store.search(test_query, k=3)
                
                print(f"\nSearch results for '{test_query}':")
                print("-" * 50)
                for i, doc in enumerate(results, 1):
                    print(f"{i}. File: {doc.metadata.get('file_name', 'Unknown')}")
                    print(f"   Content: {doc.page_content[:100]}...")
                    print()
                
                # Show collection info
                vector_store.get_collection_info()
            else:
                print("[ERROR] Failed to add documents")
        else:
            print("[ERROR] Failed to initialize vector store")
    else:
        print("[ERROR] No documents found to process")    