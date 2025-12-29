from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from config import Config
import google.generativeai as genai
from mcp_tools import FilesystemTools, ToolResult
import json
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from pathlib import Path
import subprocess


class RAGSystem:
    def __init__(self):
        self.config = Config()
        self.vector_store = None
        self.llm = None
        self.tools = FilesystemTools()
        self.retriever = None
        self.web_search = DuckDuckGoSearchRun() #free web search tool
        
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
        
    def evaluate_answer_quality(self, question: str, answer: str, sources: List[Dict])-> Dict[str, Any]:
        """ask llm to evaluate if the rag answer is good enough or if we should fallback to mcp tools"""
        evaluation_prompt = f"""You are an answer quality evaluator.

Question: {question}

Answer: {answer}

Sources: {len(sources)} document chunks

Does this answer DIRECTLY and CORRECTLY answer the user's question?
If the answer says "I don't know" or "not relevant", it's NOT satisfactory.

Return ONLY this JSON (no extra text):
{{
  "is_satisfactory": true,
  "confidence": 0.9,
  "reason": "brief explanation"
}}"""   
        try:
            response = self.llm.invoke(evaluation_prompt)
            
            # Debug: show raw response
            if not response or len(response.strip()) == 0:
                print("🐛 DEBUG - LLM returned empty response for evaluation")
                return {
                    "is_satisfactory": False,
                    "confidence": 0.0,
                    "reason": "LLM returned empty response - assuming unsatisfactory"
                }
            
            print(f"🐛 DEBUG - Raw evaluation response (first 200 chars): {response[:200]}")
            
            #clean response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1]
            
            response = response.strip()
            
            if not response:
                print("🐛 DEBUG - Response is empty after cleaning")
                return {
                    "is_satisfactory": False,
                    "confidence": 0.0,
                    "reason": "Empty response after cleaning"
                }
            
            evaluation = json.loads(response)
            print(f"🧪 Answer evaluation: {evaluation.get('reason','no reason provided')}")
            return evaluation
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error in evaluation: {e}")
            print(f"🐛 DEBUG - Attempted to parse: '{response}'")
            return {
                "is_satisfactory": False,
                "confidence": 0.0,
                "reason": "JSON parsing failed - assuming unsatisfactory"
            }
        except Exception as e:
            print(f"⚠️ Error evaluating answer quality: {e}")
            return {
                "is_satisfactory": False,
                "confidence": 0.0,
                "reason": "Error during evaluation"
            }
    def find_unprocessed_files(self)-> List[Path]:
        """find files in documents folder not yet in vector store"""
        docs_path = Path(self.config.DOCUMENTS_PATH)
        if not docs_path.exists():
            print(f'Document path {docs_path} does not exist.')
            return []
        #get all supported files
        all_files = []
        for ext in ["*.pdf", "*.docx", "*.txt", "*.md"]:
            all_files.extend(docs_path.rglob(ext))
        
        #get files already in vector store
        try:
            collection = self.vector_store.vector_store._collection
            metadata = collection.get(include=["metadatas"])
            processed_files = set(m.get('file_name') for m in metadata['metadatas'] if m.get('file_name'))
        except:
            processed_files = set()
        #find unprocessed files
        unprocessed_files = [f for f in all_files if f.name not in processed_files]
        
        if unprocessed_files:
            print(f" Found {len(unprocessed_files)} unprocessed files.")
            for f in unprocessed_files:
                print(f" - {f.name}")
        return unprocessed_files
    
    def process_new_files(self)->bool:
        """trigger document processing for new files not yet in vector store"""
        try:
            print("processing new files...")
            result = subprocess.run(
                ["uv", "run", "document_processor.py"],
                capture_output=True,
                text=True,
                timeout=200
                
            )
            if result.returncode ==0:
                print(" new files processed successfully.")
                #reinitialize retriever to get new data
                self.retriever = self.vector_store.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                return True
            else:
                print(f" Error processing new files: {result.stderr}")
                return False
        except Exception as e:
            print(f" Exception processing new files: {e}")
            return False
        
        
    def web_search_fallback(self, question: str)->str:
         """search the internet when local documents dont have the answer"""        
         try:
             print(" performing web search fallback...")
             search_results = self.web_search.run(question)
             
             #ask llm to synthesize a brief answer from search results
             synthesis_prompt = f"""Based on these web search results, answer the user's question.

Question: {question}

Search Results:
{search_results}

Provide a clear, concise answer:"""
             answer = self.llm.invoke(synthesis_prompt)
             return f"**Web Search Result:**\n\n{answer} \n\n_source : DuckDuckGo Search_"
         except Exception as e:
             print(f" Error during web search fallback: {e}")
             return " Error performing web search."  
                                            
        
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
    
    

    def query(self, question: str) -> Dict[str, Any]:
        """Main query handler with multi-stage intelligence"""
        if not self.vector_store or not self.llm:
            return {
                "answer": "❌ RAG system not initialized.",
                "sources": []
            }
        
        # Fast path: Check for simple MCP commands first
        keyword_decision = self.detect_intent_keyword(question)
        if keyword_decision:
            print("⚡ Fast keyword match - executing MCP tool directly")
            action = keyword_decision['action']
            parameter = keyword_decision['parameter']
            tool_output = self.execute_tool_action(action, parameter)
            return {
                "answer": f"**Tool Output ({action}):**\n\n{tool_output}",
                "sources": [],
                "context": "MCP Tool (keyword detection)"
            }
        
        # STAGE 1: Try RAG
        print("🔍 Stage 1: Searching course documents...")
        try:
            relevant_docs = self.retriever.invoke(question)
            print(f"📄 Retrieved {len(relevant_docs)} documents")
            
            if relevant_docs and len(relevant_docs) > 0:
                context = "\n\n".join([d.page_content for d in relevant_docs])
                prompt_template = self.create_rag_prompt()
                prompt = prompt_template.format(context=context, question=question)
                
                print("🤖 Generating answer from documents...")
                rag_answer = self.llm.invoke(prompt)
                
                print(f"🐛 DEBUG - RAG Answer preview: {rag_answer[:150]}...")
                
                sources = [
                    {
                        "file_name": doc.metadata.get("file_name", "Unknown"),
                        "chunk_index": doc.metadata.get("chunk_index", 0)
                    }
                    for doc in relevant_docs
                ]
                
                # STAGE 2: Evaluate answer quality
                print("🧪 Stage 2: Evaluating answer quality...")
                evaluation = self.evaluate_answer_quality(question, rag_answer, sources)
                
                print(f"🐛 DEBUG - Evaluation result: {evaluation}")
                
                if evaluation.get("is_satisfactory", False):
                    print("✅ Answer is satisfactory!")
                    return {
                        "answer": rag_answer,
                        "sources": sources,
                        "context": "RAG",
                        "confidence": evaluation.get("confidence", 0.0)
                    }
                else:
                    print(f"⚠️ Answer not satisfactory: {evaluation.get('reason')}")
        
        except Exception as e:
            print(f"⚠️ RAG failed: {e}")
        
        # STAGE 3: Ask LLM to check for unprocessed files using MCP
        print("🤔 Stage 3: Asking LLM to check for unprocessed files...")
        
        file_check_prompt = f"""The RAG system couldn't answer this question well: "{question}"

You have access to MCP filesystem tools. Check if there are unprocessed documents that might help.

Available tools:
- list_files: List files in documents folder
- search_files: Find files by pattern (e.g., "*.pdf")

Your task:
1. List files in the "documents" folder
2. Check if there are files that might be relevant to: "{question}"
3. Respond with JSON:
{{
  "action": "list_files" or "search_files",
  "parameter": "documents" or search pattern,
  "reasoning": "why you chose this"
}}

Return ONLY JSON:"""

        try:
            response = self.llm.invoke(file_check_prompt)
            
            # Clean response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1]
            
            decision = json.loads(response.strip())
            print(f"🔧 LLM decided: {decision.get('action')} - {decision.get('reasoning', 'No reason')}")
            
            # Execute the MCP tool the LLM chose
            action = decision.get("action")
            parameter = decision.get("parameter")
            tool_result = self.execute_tool_action(action, parameter)
            
            print(f"📂 MCP Tool Result:\n{tool_result[:200]}...")
            
            # STAGE 4: Ask LLM if it found unprocessed files
            analysis_prompt = f"""You checked the filesystem and got this result:

{tool_result}

Original question: "{question}"

Analyze:
1. Are there NEW files (not yet processed) that might answer the question?
2. Should we process them and try again?

Respond with JSON:
{{
  "has_new_files": true/false,
  "files_to_process": ["file1.pdf", "file2.docx"],
  "should_retry": true/false,
  "reasoning": "explanation"
}}

Return ONLY JSON:"""

            analysis_response = self.llm.invoke(analysis_prompt)
            
            # Clean response
            if "```json" in analysis_response:
                analysis_response = analysis_response.split("```json")[1].split("```")[0]
            elif "```" in analysis_response:
                analysis_response = analysis_response.split("```")[1]
            
            

            analysis = json.loads(analysis_response.strip())
            print(f"📊 Analysis: {analysis.get('reasoning')}")
            
            if analysis.get("has_new_files", False) and analysis.get("should_retry", False):
                print(f"🔄 Found new files: {analysis.get('files_to_process')}. Processing...")
                
                if self.process_new_files():
                    print("♻️  Retrying RAG with newly processed documents...")
                    
                    # Retry RAG
                    relevant_docs = self.retriever.invoke(question)
                    if relevant_docs and len(relevant_docs) > 0:
                        context = "\n\n".join([d.page_content for d in relevant_docs])
                        prompt_template = self.create_rag_prompt()
                        prompt = prompt_template.format(context=context, question=question)
                        
                        rag_answer = self.llm.invoke(prompt)
                        sources = [
                            {
                                "file_name": doc.metadata.get("file_name", "Unknown"),
                                "chunk_index": doc.metadata.get("chunk_index", 0)
                            }
                            for doc in relevant_docs
                        ]
                        
                        return {
                            "answer": rag_answer,
                            "sources": sources,
                            "context": "RAG (after processing new files)"
                        }
            else:
                # LLM said no new files or shouldn't retry
                # Ask if the existing files are sufficient
                print("📋 No new files to process. Checking if existing info is sufficient...")
                
                final_decision_prompt = f"""The filesystem check showed:
{tool_result}

Analysis: {analysis.get('reasoning')}

Original question: "{question}"

Should we:
1. Web search (if we have NO relevant local files)
2. Stop here (if we found files but they're already processed, or question is not relevant to documents)

Return JSON:
{{
  "action": "web_search" or "stop",
  "reasoning": "brief explanation"
}}

Return ONLY JSON:"""

                final_response = self.llm.invoke(final_decision_prompt)
                
                # Clean response
                if "```json" in final_response:
                    final_response = final_response.split("```json")[1].split("```")[0]
                elif "```" in final_response:
                    final_response = final_response.split("```")[1]
                
                final_decision = json.loads(final_response.strip())
                print(f"🎯 Final decision: {final_decision.get('action')} - {final_decision.get('reasoning')}")
                
                if final_decision.get("action") == "stop":
                    return {
                        "answer": f"❌ Unable to answer this question with available resources.\n\n{final_decision.get('reasoning')}",
                        "sources": [],
                        "context": "No suitable data"
                    }
                # If action is "web_search", continue to Stage 5 below
        
        except Exception as e:
            print(f"⚠️ File check failed: {e}")
        
        # STAGE 5: Web search fallback (only if we get here)
        print("🌐 Stage 5: Searching the web...")
        web_answer = self.web_search_fallback(question)
        
        return {
            "answer": web_answer,
            "sources": [],
            "context": "Web Search"
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