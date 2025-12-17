from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from config import Config
import google.generativeai as genai
from mcp_tools import MCPTools, MCPToolResult
import json
from langchain_community.llms import ollama


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
            # Try Google embeddings first, fallback to local
            if not self.vector_store.initialize(use_local_embeddings=False):
                print(" Failed to initialize with Google embeddings, trying local...")
                if not self.vector_store.initialize(use_local_embeddings=True):
                    print(" Failed to initialize vector store.")
                    return False
            
            # Set up retriever
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            
            
            # Initialize Gemini LLM
            print(" Initializing Gemini LLM...")
            genai.configure(api_key=self.config.GOOGLE_API_KEY)
            self.llm = GoogleGenerativeAI(
                model=f"models/{self.config.LLM_MODEL}",
                google_api_key=self.config.GOOGLE_API_KEY,
                temperature=0.1
            )
            
            print(" Simple RAG system initialized successfully!")
            return True
            
        except Exception as e:
            print(f" Error initializing RAG system: {e}")
            return False
        
        
        
    def detect_action(self, question: str)->Dict[str, Any]:
        """ask llm to decide whether to use mcp tool or RAG"""
        router_template = """
        You are an intelligent router for a student assistant. 
        You have access to the following tools:
        1. list_files: List files in a directory (use for "ls", "show files", "what files do I have")
        2. read_file: Read content of a specific file (use for "read", "cat", "open", "display")
        3. search_files: Search for files by name (use for "find", "search", "locate")
        4. file_info: Get metadata about a file (use for "size", "info", "created date")
        5. rag_search: Search the vector database for course content (use for questions about concepts, definitions, summaries)

        User Question: {question}

        Return a JSON object with two keys: "action" and "parameter".
        - "action": One of ["list_files", "read_file", "search_files", "file_info", "rag_search"]
        - "parameter": The argument for the tool (e.g., filename, search term). For "rag_search", return the original question.
        - For "list_files", the parameter is usually "." unless a folder is specified.

        JSON Response:
        """
        prompt = PromptTemplate(template=router_template, input_variables=["question"])
        formatted_prompt = prompt.format(question=question)
        
        try:
            #get response from llm
            response_text = self.llm.invoke(formatted_prompt)
            
            #clean up response to extract JSON
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            decision = json.loads(response_text)
            return decision
        except Exception as e:
            print(f" Error detecting action: {e}, defaulting to RAG search.")
            return {"action": "rag_search", "parameter": question}
        
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
        if not self.vector_store or not self.llm:
            return{
                "answer": " RAG system not initialized. Call initialize() first.",
                "sources": []
            }                        
        # 1-ask llm what to do 
        print(f"thinking about how to handle the question...:{question}")
        decision = self.detect_action(question)
        action = decision.get("action", "rag_search")
        parameter = decision.get("parameter")
        
        print (f" decided action: {action} with parameter: {parameter}")
        
        # 2- execute tool or RAG based on decision
        if action !="rag_search":
            tool_output = self.execute_tool_action(action, parameter)
            return {
                "answer": f"**TOOL OUTPUT({action}):** \n\n{tool_output}",
                "sources": [],
                "context": "Tool execution"
            }
        #3- standard RAG flow
        try:
            print(f"seaching documents for RAG...")
            relevant_docs = self.retriever.invoke(question)
            if not relevant_docs:
                return {
                    "answer": " Aucun document pertinent trouvé pour cette question.",
                    "sources": []
                }
            context = "\n\n".join([
                d.page_content for d in relevant_docs
            ])
            prompt = self.create_rag_prompt().format(context=context, question=question)
            print(f" generating answer with Gemini...")
            response = self.llm.invoke(prompt)
            sources = [
                {"filename": doc.metadata.get('file_name', 'Unknown'),
                 "chunk_id": doc.metadata.get('chunk_index', 0)} for doc in relevant_docs  
            ] 
            return {
                "answer": response,
                "sources": sources
            }           
        except Exception as e:
            return {
                "answer": f" Erreur lors du traitement: {e}",
                "sources": []
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    def execute_tool(self, question: str) -> str:
        """execute MCP tool based on detected intent"""
        # Use original case for filename extraction
        question_original = question
        question_lower = question.lower()
        
        try:
            # LIST DIRECTORY
            # Matches: "list files", "ls " (start), "ls" (exact), "show files"
            if any(k in question_lower for k in ["list files", "list directory", "show files", "what files"]) or \
               question_lower.startswith("ls ") or question_lower == "ls":
                
                path = "."
                if "in " in question_lower:
                    parts = question_lower.split("in ")
                    if len(parts) > 1:
                        path = parts[1].strip().split()[0]
                
                result = self.tools.list_directory(path)
                return result.content if result.success else f"Error: {result.error}"

            # READ FILE
            # Matches: "read ...", "cat ...", "open ...", "display ...", "... content"
            elif any(k in question_lower for k in ["read ", "cat ", "open ", "display "]) or "content" in question_lower:
                
                # Extract filename using original case
                words = question_original.split()
                filename = words[-1] # Default to last word
                
                # Try to find filename with extension
                for word in words:
                    if "." in word and len(word) > 2:
                        filename = word
                        break
                
                result = self.tools.read_file(filename)
                return result.content if result.success else f"Error: {result.error}"
            
            # SEARCH FILES
            # Matches: "search ...", "find ..."
            elif "search" in question_lower or "find" in question_lower:
                pattern = ""
                if "search for" in question_lower:
                    parts = question_lower.split("search for")
                    if len(parts) > 1: pattern = parts[1].strip().split()[0]
                elif "find" in question_lower:
                    parts = question_lower.split("find")
                    if len(parts) > 1: pattern = parts[1].strip().split()[0]
                elif "search" in question_lower:
                    parts = question_lower.split("search")
                    if len(parts) > 1: pattern = parts[1].strip().split()[0]
                
                if not pattern:
                    return "Error: Could not determine search pattern."

                result = self.tools.search_files(pattern)
                return result.content if result.success else f"Error: {result.error}"
            
            # FILE INFO
            # Matches: "file info", "file size", "info ..."
            elif any(k in question_lower for k in ["info", "size", "created"]):
                words = question_original.split()
                filename = words[-1]
                for word in words:
                    if "." in word and len(word) > 2:
                        filename = word
                        break
                result = self.tools.get_file_info(filename)
                return result.content if result.success else f"Error: {result.error}"

            else:
                return "Commande non reconnue pour les outils MCP."

        except Exception as e:
            return f"Erreur lors de l'exécution de l'outil: {e}"    
        
    def create_prompt_template(self):
        """Create a prompt template for question answering"""
        template = """
Tu es un assistant intelligent qui aide les étudiants avec leurs documents de cours.
Utilise le contexte fourni pour répondre à la question de manière précise et utile.

Contexte:
{context}

Question: {question}

Instructions:
- Réponds en français si la question est en français, en anglais si elle est en anglais
- Base ta réponse uniquement sur le contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois précis et structuré dans ta réponse
- Cite les sources quand c'est possible

Réponse:
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a question and return an answer with sources"""
        if not self.vector_store or not self.llm:
            return {
                "answer": " RAG system not initialized. Call initialize() first.",
                "sources": [],
                "error": "Not initialized"
            }
        
        # Check intent
        intent = self.detect_intent(question)
        
        if intent == "tool":
            print(f" Detected tool request...")
            tool_output = self.execute_tool(question)
            return {
                "answer": f"**Tool Execution Result:**\n\n{tool_output}",
                "sources": [],
                "context": "Tool execution"
            }

        # Standard RAG flow
        try:
            # Retrieve relevant documents
            print(f"🔍 Searching for relevant documents...")
            relevant_docs = self.retriever.invoke(question)
            
            if not relevant_docs:
                return {
                    "answer": " Aucun document pertinent trouvé pour cette question.",
                    "sources": [],
                    "error": "No relevant documents"
                }
            
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Source: {doc.metadata.get('file_name', 'Unknown')}\n{doc.page_content}"
                for doc in relevant_docs
            ])
            
            # Create prompt
            prompt_template = self.create_prompt_template()
            prompt = prompt_template.format(context=context, question=question)
            
            # Generate answer
            print(f" Generating answer with Gemini...")
            response = self.llm.invoke(prompt)
            
            # Extract sources
            sources = [
                {
                    "filename": doc.metadata.get('file_name', 'Unknown'),
                    "chunk_id": doc.metadata.get('chunk_index', 0),
                    "content_preview": doc.page_content[:200] + "..."
                }
                for doc in relevant_docs
            ]
            
            return {
                "answer": response,
                "sources": sources,
                "context": context[:500] + "..." if len(context) > 500 else context
            }
            
        except Exception as e:
            return {
                "answer": f" Erreur lors du traitement: {e}",
                "sources": [],
                "error": str(e)
            }
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n🎓 Smart Notes Assistant - Mode Interactif")
        print("=" * 60)
        print("Posez vos questions sur vos documents de cours!")
        print("Commandes MCP disponibles:")
        print("- 'list files': Voir les fichiers")
        print("- 'read [filename]': Lire un fichier")
        print("- 'search [pattern]': Chercher un fichier")
        print("- 'quit' ou 'exit': Quitter")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Votre question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Au revoir!")
                    break
                
                if not question:
                    continue
                
                print("\n Traitement...")
                result = self.query(question)
                print(f"\n **********Réponse:***********")
                print("-" * 40)
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\n📚 **Sources consultées:**")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"{i}. {source['filename']} (chunk {source['chunk_id']})")
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    rag = RAGSystem()
    if rag.initialize():
        rag.interactive_chat()
    else:
        print("❌ Échec de l'initialisation du système RAG")