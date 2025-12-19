from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from config import Config
import google.generativeai as genai
from mcp_tools import MCPTools, MCPToolResult
import json
from langchain_community.llms import Ollama


class RAGSystem:
    def __init__(self):
        self.config = Config()
        self.vector_store = None
        self.llm = None
        self.tools = MCPTools()
        self.retriever = None
        
    def initialize(self):
        """Initialize the RAG system with vector store and LLM"""
        try:
            # Initialize vector store
            print(" Initializing vector store...")
            self.vector_store = VectorStore()
           
            #use local embeddings if  config says so 
            use_local = self.config.USE_LOCAL_LLM
            print(f"  Using local embeddings: {use_local}")
            
            # Try Google embeddings first, fallback to local
            if not self.vector_store.initialize(use_local_embeddings=use_local):
                print(" Failed to initialize with Google embeddings, trying local...")
                return False
            
            # Set up retriever
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            
            # ----LLM Initialization----
            if self.config.USE_LOCAL_LLM:
                print(f"🔄 Connecting to Local Ollama ({self.config.OLLAMA_MODEL})...")
                try:
                    self.llm = Ollama(
                        model=self.config.OLLAMA_MODEL,
                        temperature=0.1
                    )
                    #test connection
                    self.llm.invoke("Hello")
                    print("✓ Connected to local Ollama LLM successfully.")
                except Exception as e:
                    print(f" Error connecting to Ollama LLM: {e}")
                    return False
            else:
                print(" connecting to Gemini LLM...")
                genai.configure(api_key=self.config.GOOGLE_API_KEY)
                self.llm = GoogleGenerativeAI(
                    model=f"models/{self.config.LLM_MODEL}",
                    google_api_key=self.config.GOOGLE_API_KEY,  
                    temperature=0.1
                )
                print("✓ Connected to Gemini LLM successfully.")
            return True    
          
        except Exception as e:
            print(f" Error initializing RAG system: {e}")
            return False
        
        
        
    def detect_intent_keyword(self, question: str) -> Dict[str, str]:
        """Fast, free keyword detection to save time/quota"""
        q = question.lower()
        
        # List Files
        if any(k in q for k in ["list files", "list directory", "show files", "ls "]) or q == "ls":
            path = "."
            if "in " in q:
                parts = q.split("in ")
                if len(parts) > 1: path = parts[1].strip().split()[0]
            return {"action": "list_files", "parameter": path}

        # Read File
        if any(k in q for k in ["read ", "cat ", "open "]):
            words = question.split()
            filename = words[-1]
            for word in words:
                if "." in word and len(word) > 2: filename = word; break
            return {"action": "read_file", "parameter": filename}

        # Search Files
        if "search for" in q or "find file" in q:
            term = q.split("for")[-1].strip() if "for" in q else q.split("file")[-1].strip()
            return {"action": "search_files", "parameter": term}

        return None 

    def decide_action(self, question: str) -> Dict[str, Any]:
        """Ask LLM to decide which MCP tool to use (only called after RAG fails)"""
        
        # 1. Try fast keyword match first
        keyword_decision = self.detect_intent_keyword(question)
        if keyword_decision:
            print("⚡ Fast keyword match used")
            return keyword_decision

        # 2. Fallback to LLM Router
        print("🤔 Asking LLM which filesystem tool to use...")
        router_template = """The user asked a question, but we found nothing in the course documents.
Now we need to search the filesystem.

Available tools:
- list_files: List all files in current directory or a path
- search_files: Find files matching a pattern (e.g., "*.py" for Python files, "vector" for filenames containing "vector")
- read_file: Read content of a specific file
- file_info: Get file metadata (size, date)

Question: {question}

Examples:
- "show me python files" → {{"action": "search_files", "parameter": "*.py"}}
- "find config file" → {{"action": "search_files", "parameter": "config"}}
- "list all files" → {{"action": "list_files", "parameter": "."}}
- "read README" → {{"action": "read_file", "parameter": "README.md"}}

Return ONLY JSON with "action" and "parameter":"""

        prompt = PromptTemplate(template=router_template, input_variables=["question"])
        formatted_prompt = prompt.format(question=question)
        
        try:
            response_text = self.llm.invoke(formatted_prompt)
            
            print(f"🐛 DEBUG - Raw LLM response: {response_text[:200]}...")
            
            # Clean up response
            if not response_text or response_text.strip() == "":
                print("⚠️ Empty response, defaulting to list_files")
                return {"action": "list_files", "parameter": "."}
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1]
            
            response_text = response_text.strip()
            print(f"🐛 DEBUG - Cleaned response: {response_text}")
            
            decision = json.loads(response_text)
            return decision
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}, defaulting to list_files")
            return {"action": "list_files", "parameter": "."}
        except Exception as e:
            print(f"⚠️ Router error: {e}, defaulting to list_files")
            return {"action": "list_files", "parameter": "."}        
        
    def execute_tool_action(self, action:str, parameter:str)->str:
        """execute the tool decided by the LLM"""
        try:
            if action == "list_files":
                result= self.tools.list_directory(parameter if parameter else ".")
            elif action == "read_file":
                result= self.tools.read_file(parameter)
            elif action == "search_files":
                result= self.tools.search_files(parameter)
            elif action == "file_info":
                result= self.tools.get_file_info(parameter)
            else:
                return "Invalid tool action."
            
            return result.content if result.success else f"Error: {result.error}"
        except Exception as e:
            return f"Error executing tool action: {e}"
        
    def create_rag_prompt(self):
        """crreate prompt template for RAG"""
        template = """
        tu es un assistant intelligent qui aide les étudiants avec leurs documents de cours.
        Utilise le contexte fourni pour répondre à la question de manière précise et utile.
        
        Contexte:
        {context}
        question: 
        {question}
        reponse:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def query(self,question: str)->Dict[str, Any]:
        """Main query handler: Try RAG first, fallback to MCP tools"""
        if not self.vector_store or not self.llm:
            return{
                "answer": "❌ RAG system not initialized. Call initialize() first.",
                "sources": []
            }                        
        
        # STEP 1: Always try RAG first
        print("🔍 Searching course documents...")
        try:
            # Debug: Check if retriever exists and vector store has data
            print(f"🐛 DEBUG - Retriever exists: {self.retriever is not None}")
            if self.vector_store:
                collection = self.vector_store.vector_store._collection
                print(f"🐛 DEBUG - Documents in vector store: {collection.count()}")
            
            relevant_docs = self.retriever.invoke(question)
            print(f"🐛 DEBUG - Retrieved {len(relevant_docs)} documents")
            
            if relevant_docs and len(relevant_docs) > 0:
                # Found documents! Generate answer from them
                context = "\n\n".join([d.page_content for d in relevant_docs])
                prompt_template = self.create_rag_prompt()
                prompt = prompt_template.format(context=context, question=question)
                
                print(f"✅ Found {len(relevant_docs)} relevant documents. Generating answer...")
                response = self.llm.invoke(prompt)
                
                sources = [
                    {"filename": doc.metadata.get('file_name', 'Unknown'),
                     "chunk_id": doc.metadata.get('chunk_index', 0)} 
                    for doc in relevant_docs
                ]
                
                return {
                    "answer": response,
                    "sources": sources,
                    "context": "RAG"
                }
        except Exception as e:
            print(f"⚠️ RAG search error: {e}")
        
        # STEP 2: RAG found nothing or failed - Try MCP tools
        print("⚠️ No relevant documents found. Checking filesystem with MCP tools...")
        
        # Use the router to decide which tool
        decision = self.decide_action(question)
        action = decision.get("action", "rag_search")
        parameter = decision.get("parameter")
        
        print(f"👉 Decided action: {action} with parameter: {parameter}")
        
        # If LLM still says "rag_search", we're out of options
        if action == "rag_search":
            return {
                "answer": "❌ Je n'ai trouvé aucune information sur ce sujet dans vos documents ni dans vos fichiers.",
                "sources": [],
                "context": "No results"
            }
        
        # Execute the MCP tool
        tool_output = self.execute_tool_action(action, parameter)
        return {
            "answer": f"**Tool Output ({action}):**\n\n{tool_output}",
            "sources": [],
            "context": "Tool execution"
        }                
    def interactive_chat(self):
        """start interactive chat session"""
        print("\n Smart Notes Assistant - Interactive Mode")
        print("="*60)
        print("Ask your questions about your course documents!")
        print("Available MCP commands:")
        print("- 'list files': View files")
        print("- 'read [filename]': Read a file")
        print("- 'search [pattern]': Search for a file")
        print("- 'quit' or 'exit': Exit")
        print("="*60)
        while True:
            try:
                question = input("\n Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print(" Goodbye!")
                    break
                if not question:
                    continue
                result = self.query(question)
                print(f"\n **Response:**\n{'-'*40}\n{result['answer']}\n{'-'*40}") 
            except KeyboardInterrupt:
                print("\n Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")
                    
        
        
        
   

if __name__ == "__main__":
    rag = RAGSystem()
    if rag.initialize():
        rag.interactive_chat()
    else:
        print("❌ Échec de l'initialisation du système RAG")