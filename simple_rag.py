from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from config import Config
import google.generativeai as genai
from mcp_tools import MCPTools, MCPToolResult

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
                print("⚠️ Failed to initialize with Google embeddings, trying local...")
                if not self.vector_store.initialize(use_local_embeddings=True):
                    print("❌ Failed to initialize vector store.")
                    return False
            
            # Set up retriever
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            
            
            # Initialize Gemini LLM
            print("🔄 Initializing Gemini LLM...")
            genai.configure(api_key=self.config.GOOGLE_API_KEY)
            self.llm = GoogleGenerativeAI(
                model=f"models/{self.config.LLM_MODEL}",
                google_api_key=self.config.GOOGLE_API_KEY,
                temperature=0.3
            )
            
            print("✅ Simple RAG system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
        
        
        
    def detect_intent(self, question: str) -> str:
        """simple intent detection using keywords"""
        question_lower = question.lower()
        
        # Tool keywords
        tool_keywords = [
            "list files", "list directory", "show files", "what files", "ls ",
            "read file", "read content", "open file", "display file", "read ", "cat ",
            "search file", "find file", "search for", "find ", "search ",
            "file info", "file size", "when was created", "info ",
            "run command", "execute command", "system info"
        ]
        
        # print(f"DEBUG: Checking intent for '{question_lower}'")
        for keyword in tool_keywords:
            if keyword in question_lower:
                # print(f"DEBUG: Matched keyword '{keyword}'")
                return "tool"
                
        return "rag"
    
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