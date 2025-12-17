import os 
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

env_path = Path(__file__).parent / ".env"

load_dotenv(env_path)

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./documents")
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL_NAME = "models/text-embedding-004"
    LLM_MODEL = "gemini-2.0-flash"  # Fixed model name
    
    # Local embedding model as fallback
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    
    #init embedding model
    _embedding_model = None
    _local_embedding_model = None
    
    @classmethod
    def get_embedding_model(cls, use_local=False):
        """Get embedding model - local or API-based"""
        if use_local:
            if cls._local_embedding_model is None:
                cls._local_embedding_model = HuggingFaceEmbeddings(
                    model_name=cls.LOCAL_EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                print("✓ Local embedding model initialized")
            return cls._local_embedding_model
        else:
            if cls._embedding_model is None:
                cls.validate()
                cls._embedding_model = GoogleGenerativeAIEmbeddings(
                    model=cls.EMBEDDING_MODEL_NAME,
                    google_api_key=cls.GOOGLE_API_KEY
                )
            return cls._embedding_model    
    
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
        
        #configure the API key for google generative AI
        genai.configure(api_key=cls.GOOGLE_API_KEY)
        #create necessary directories
        Path(cls.CHROMA_DB_PATH).mkdir( exist_ok=True)
        Path(cls.DOCUMENTS_PATH).mkdir( exist_ok=True)
        
        print(f"✓ Directories created:")
        print(f"  - Chroma DB Path: {cls.CHROMA_DB_PATH}")
        print(f"  - Documents Path: {cls.DOCUMENTS_PATH}")
        
        
        
if __name__ == "__main__":
    try:
        print("Validating configuration...")
        Config.validate()
        print("✓ Configuration is valid.")
        print(f"✓ Using LLM model: {Config.LLM_MODEL}")
        print(f"✓ Using Embedding model: {Config.EMBEDDING_MODEL_NAME}")
        
        #test embedding model initialization (without actual embedding)
        embedding_model = Config.get_embedding_model()
        print("✓ Embedding model initialized successfully.")
        
        print("\n Configuration test passed! Ready to proceed to next step.")
        print(" Note: Skipping embedding test due to API quota limits.")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print("Please check your .env file and API key.")