from typing import List, Optional
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import Config
from document_processor import DocumentProcessor
import os 


class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embedding_model = None
        self.vector_store = None
        self.collection_name = "smart_notes_collection"
        
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
            print("✓ Vector store initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]):
        """add documents to the vector store"""
        if self.vector_store is None:
            print("Vector store is not initialized.")
            return False
        
        try:
            print(f"adding {len(documents)} documents to vector store...")
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            print("documents added and vector store persisted")
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
        
    def search(self,query: str, k: int =5) -> List[Document]:
        """perform similarity search in the vector store"""
        if self.vector_store is None:
            print("Vector store is not initialized.")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            print(f"found {len(results)} similar documents for query: {query}")
            return results
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    
    def get_collection_info(self):
        """get info about the chroma collection"""
        if self.vector_store:
            try:
                collection = self.vector_store._collection
                count = collection.count()
                print(f"Collection '{self.collection_name}' has {count} items.")
                return count
            except Exception as e:
                print(f"Error getting collection info: {e}")
                return 0
        return 0        
    
    
#test vector store fuctionality

if __name__ == "__main__":
    print("🔄 Setting up Smart Notes Vector Store...")
    
    processor = DocumentProcessor()
    documents = processor.process_documents()
    if documents:
        print(f"✓ Processed {len(documents)} documents.")
        
        # Initialize vector store with local embeddings
        vector_store = VectorStore()
        
        print("\n🚀 Initializing vector store with local embeddings...")
        success = vector_store.initialize(use_local_embeddings=True)
        
        if success:
            print("📥 Adding documents to vector store...")
            add_success = vector_store.add_documents(documents)
            
            if add_success:
                print("✅ Vector store setup complete!")
                
                # Test search functionality
                print("\n🔍 Testing search functionality...")
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
                print("❌ Failed to add documents")
        else:
            print("❌ Failed to initialize vector store")
    else:
        print("❌ No documents found to process")    