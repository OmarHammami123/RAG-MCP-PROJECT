import os 
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config    


class DocumentProcessor:
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    def load_documents(self, file_path : Path)-> str:
        """load content from different file types"""
        file_extension = file_path.suffix.lower()
        try:
            if file_extension == ".pdf":
                return self._load_pdf(file_path)
            elif file_extension == ".docx":
                return self._load_docx(file_path)
            elif file_extension in ['.txt', '.md']:
                return self._load_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            return ""
        
        
    def _load_pdf(self, file_path: Path) -> str:
        """load content from PDF file  """   
        text = ""
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    
    def _load_docx(self, file_path: Path) -> str:
        """load content from DOCX file  """   
        return docx2txt.process(file_path)
    
    def _load_txt(self, file_path: Path) -> str:
        """load content from TXT or MD file  """   
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
        
        
    def process_documents(self, document_path: str = None) -> List[Document]:
        """process all documents in the document path"""
        if document_path is None :
            document_path = self.config.DOCUMENTS_PATH
        
        documents = []
        docs_path = Path(document_path)
        
        if not docs_path.exists():
            print(f'Document path {docs_path} does not exist.')
            return documents
        
        supported_extensions = [".pdf", ".docx", ".txt", ".md"]
        
        for file_path in docs_path.rglob("*"):
          if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
              print(f"Processing file: {file_path}")
              
              #load document content
              content = self.load_documents(file_path)
              
              if content.strip():
                  #split content into chunks
                  chunks = self.text_splitter.split_text(content)
                  
                  #create doc objects
                  for i, chunk in enumerate(chunks):
                      doc = Document(
                          page_content = chunk,
                          metadata = {
                              "source": str(file_path),
                              "file_name": file_path.name,
                              "chunk_index": i,
                              "total_chunks": len(chunks)
                          }
                      )
                      documents.append(doc)
        print(f" Processed {len(documents)} document chunks from {len(set(doc.metadata['file_name'] for doc in documents))} files")
        return documents              
    
    
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    documents = processor.process_documents()
    
    if documents:
        print("\n📄 Sample document chunk:")
        print("-" * 40)
        print(f"Content: {documents[0].page_content[:300]}...")
        print(f"\nMetadata: {documents[0].metadata}")
        print(f"\n📊 Total files processed: {len(set(doc.metadata['file_name'] for doc in documents))}")
        print(f"📊 Total chunks created: {len(documents)}")
    else:
        print("No documents were processed. Make sure you have PDF, DOCX, TXT, or MD files in the documents folder.")      