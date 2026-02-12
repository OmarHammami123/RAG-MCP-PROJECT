import time
from caching import QueryCache
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from vector_store import VectorStore
from config import Config
import google.generativeai as genai

import json
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from pathlib import Path
import subprocess
import sys
import atexit
import os

import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client 
import threading

from logging_config import rag_logger
from exceptions import (
    RAGException, LLMException, LLMConnectionError, LLMResponseError,
    MCPException, MCPConnectionError, MCPToolError,
    DocumentProcessingException, handle_exception
)
import re
class RAGSystem:
    def __init__(self):
        self.config = Config()
        self.vector_store = None
        self.llm = None
        
        self.retriever = None
        self.web_search = DuckDuckGoSearchRun() #free web search tool
        #adding mcp client connection
        self.mcp_session = None
        self.mcp_read = None
        self.mcp_write = None
        self.mcp_client_context = None
        self.mcp_session_context = None
        
        #init query cache
        self.query_cache = QueryCache(
            ttl_minutes=60,
            cache_dir="cache",
            persist=True,
            max_entries=1000
        )
        
        # Initialize logger
        self.logger = rag_logger
        self.logger.info("RAG System initialized")
        
        # Background processes
        self.ollama_process = None
        self.mcp_server_process = None
        
        # Keep event loop alive in background thread
        self.loop = None
        self.loop_thread = None
        
    def start_ollama(self):
        """Start Ollama server in background"""
        try:
            # Check if Ollama is already running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0:
                self.logger.info("Ollama is already running")
                return True
        except:
            pass
        
        # Start Ollama serve in background
        try:
            self.logger.info("Starting Ollama server in background...")
            
            if sys.platform == "win32":
                # Windows: use START command to launch in new window
                self.ollama_process = subprocess.Popen(
                    ["cmd", "/c", "start", "/min", "ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS
                )
            else:
                # Linux/Mac: use nohup
                self.ollama_process = subprocess.Popen(
                    ["nohup", "ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=lambda: os.setpgrp()
                )
            
            # Give Ollama time to start (non-blocking check)
            self.logger.info("Waiting for Ollama to start...")
            for i in range(10):  # Try for 5 seconds
                time.sleep(0.5)
                try:
                    check = subprocess.run(
                        ["ollama", "list"],
                        capture_output=True,
                        timeout=1,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    )
                    if check.returncode == 0:
                        self.logger.info("Ollama server started successfully")
                        return True
                except:
                    continue
            
            self.logger.warning("Ollama may still be starting...")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to start Ollama: {e}. Continuing anyway...")
            return False
    
    def stop_ollama(self):
        """Stop Ollama server"""
        if self.ollama_process:
            try:
                self.logger.info("Stopping Ollama server...")
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
                self.logger.info("Ollama server stopped")
            except Exception as e:
                self.logger.warning(f"Failed to stop Ollama gracefully: {e}")
                try:
                    self.ollama_process.kill()
                except:
                    pass
    
    async def _init_mcp(self):
        """initialize mcp client connection"""
        self.logger.debug("Initializing MCP connection...")
        try:
            server_params = StdioServerParameters(
                command="uv",
                args=["run", "mcp_server.py"]
            )
            #store the context managers
            self.mcp_client_context = stdio_client(server_params)
            self.mcp_read, self.mcp_write = await self.mcp_client_context.__aenter__()
            
            self.mcp_session_context = ClientSession(self.mcp_read, self.mcp_write)
            self.mcp_session = await self.mcp_session_context.__aenter__()
            
            await self.mcp_session.initialize()
            print("[OK] MCP client connected successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Error initializing MCP client: {e}")
            import traceback
            traceback.print_exc()
            self.mcp_session = None
            return False
    
    def _start_event_loop(self):
        """Start event loop in background thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    
    async def _cleanup_mcp(self):
        """ cleanup mcp client connection"""
        if self.mcp_session:
            await self.mcp_session_context.__aexit__(None, None, None)
            await self.mcp_client_context.__aexit__(None, None, None)
        
        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
    
    
    def close(self):
        """Explicitly close the system"""
        self.logger.info("Shutting down RAG system...")
        
        if self.loop and self.loop.is_running():
            # Submit cleanup to the running loop
            future = asyncio.run_coroutine_threadsafe(self._cleanup_mcp(), self.loop)
            try:
                future.result(timeout=5)
            except:
                pass
            
            # Stop the loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            
            # Wait for thread
            if self.loop_thread:
                self.loop_thread.join(timeout=5)
        
        # Stop Ollama if we started it
        if self.config.USE_LOCAL_LLM:
            self.stop_ollama()
                
    def __del__(self):
        """cleanup on deletion"""
        try:
            self.close()
        except:
            pass
    
    
    
    def execute_tool_action(self, action: str, parameter: str) -> str:
        """execute mcp tool via async call"""
        if not self.mcp_session:
            return "[ERROR] Error: MCP session not initialized. MCP server may not be running."
        
        if not self.loop or not self.loop.is_running():
            return "[ERROR] Error: Event loop not running"
        
        try:
            # Submit the coroutine to the running event loop
            future = asyncio.run_coroutine_threadsafe(
                self._execute_mcp_tool(action, parameter),
                self.loop
            )
            # Wait for result with timeout
            result = future.result(timeout=10)
            return result
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"[ERROR] Error executing MCP tool: {e}\n\nDetails:\n{error_details}"
        
    async def _execute_mcp_tool(self, action: str, parameter: str) -> str:
        """execute mcp tool asynchronously """
        try:
            if action == "list_files":
                result = await self.mcp_session.call_tool("list_directory", {"path": parameter if parameter else "."})
                return result.content[0].text
            elif action == "read_file":
                result = await self.mcp_session.call_tool("read_file", {"path": parameter})
                return result.content[0].text
            
            elif action == "search_files":
                result = await self.mcp_session.call_tool(
                    "search_files",
                    {
                        "pattern": parameter,
                    } 
                )
                return result.content[0].text
            
            elif action == "get_file_info":
                result = await self.mcp_session.call_tool("get_file_info", {"path": parameter})
                return result.content[0].text
            
            # Write operations require permission
            elif action == "create_directory":
                result = await self.mcp_session.call_tool("create_directory", {"path": parameter})
                return result.content[0].text
            
            elif action == "write_file":
                parts = parameter.split("|")
                if len(parts) == 2:
                    result = await self.mcp_session.call_tool("write_file", {"path": parts[0], "content": parts[1]})
                    return result.content[0].text
                return "[ERROR] Error: Invalid write_file format"
            
            elif action == "move_file":
                parts = parameter.split("|")
                if len(parts) == 2:
                    result = await self.mcp_session.call_tool("move_file", {"source": parts[0], "destination": parts[1]})
                    return result.content[0].text
                return "[ERROR] Error: Invalid move_file format"
            
            elif action == "delete_file":
                result = await self.mcp_session.call_tool("delete_file", {"path": parameter})
                return result.content[0].text
            
            else:
                return "Invalid tool action."
        except Exception as e:
            return f"Error executing MCP tool: {e}"
                        
        
        
    def initialize(self):
        """Initialize the RAG system with vector store and LLM"""
        self.logger.info("Starting RAG system initialization...")
        start_time = time.time()
        try:
            # Start Ollama if using local LLM
            if self.config.USE_LOCAL_LLM:
                self.logger.info("Starting Ollama server (local LLM mode)...")
                self.start_ollama()
            
            # Start event loop in background thread for MCP
            self.logger.debug("Starting async event loop...")
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
            self.loop_thread.start()
            
            # Initialize MCP client in the background loop
            self.logger.debug("Initializing MCP client...")
            future = asyncio.run_coroutine_threadsafe(self._init_mcp(), self.loop)
            try:
                success = future.result(timeout=10)
                if not success:
                    print("[WARN] MCP initialization failed, continuing without MCP")
            except Exception as e:
                print(f"[WARN] MCP initialization timed out or failed: {e}")
            
            # Initialize vector store
            print(" Initializing vector store...")
            self.vector_store = VectorStore()
            
            #use local embeddings if config says so 
            use_local = self.config.USE_LOCAL_LLM
            print(f"  Using local embeddings: {use_local}")
            
            # Try Google embeddings first, fallback to local
            if not self.vector_store.initialize(use_local_embeddings=use_local):
                print(" Failed to initialize... trying local embeddings fallback")
                if not self.vector_store.initialize(use_local_embeddings=True):
                    print("[ERROR] Failed to initialize vector store")
                    return False
            
            # Set up retriever
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 10}  # Increased to get more context
            )
            
            # ----LLM Initialization----
            if self.config.USE_LOCAL_LLM:
                print(f"[RELOAD] Connecting to Local Ollama ({self.config.OLLAMA_MODEL})...")
                try:
                    self.llm = Ollama(
                        model=self.config.OLLAMA_MODEL,
                        temperature=0.1
                    )
                    #test connection
                    self.llm.invoke("Hello")
                    print("[OK] Connected to local Ollama LLM successfully.")
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
                print("[OK] Connected to Gemini LLM successfully.")
            return True    
          
        except Exception as e:
            print(f" Error initializing RAG system: {e}")
            return False
        
    def evaluate_answer_quality(self, question: str, answer: str, sources: List[Dict])-> Dict[str, Any]:
        """ask llm to evaluate if the rag answer is good enough or if we should fallback to mcp tools"""
        evaluation_prompt = f"""Tu es un évaluateur de qualité de réponses RAG.

Question: {question}

Réponse: {answer}

Sources: {len(sources)} chunks de documents

RÈGLES D'ÉVALUATION:

✅ MARQUER COMME SATISFAISANT si:
1. La réponse contient des FAITS CONCRETS, définitions, ou explications spécifiques
2. La réponse cite ou paraphrase le contenu des documents source
3. La réponse utilise "selon le document", "d'après le contexte", "le chapitre X indique" (BONNE PRATIQUE !)
4. La réponse fournit des chiffres, noms, dates, ou détails précis

❌ MARQUER COMME NON SATISFAISANT SEULEMENT si:
1. La réponse dit explicitement "Je ne trouve pas", "Aucune information", "Pas mentionné dans les documents"
2. La réponse est vague et ne contient AUCUN fait concret
3. La réponse invente de toute pièce sans référencer les documents
4. La réponse est un simple "je ne sais pas"

IMPORTANT: Une réponse qui cite les sources ("selon...", "d'après...") est EXCELLENTE et doit être marquée satisfaisante !

Retourne UNIQUEMENT ce JSON:
{{
  "is_satisfactory": true,
  "confidence": 0.9,
  "reason": "La réponse contient des faits spécifiques tirés des documents"
}}"""   
        try:
            response = self.llm.invoke(evaluation_prompt)
            
            # Debug: show raw response
            if not response or len(response.strip()) == 0:
                print("[DEBUG] DEBUG - LLM returned empty response for evaluation")
                return {
                    "is_satisfactory": False,
                    "confidence": 0.0,
                    "reason": "LLM returned empty response - assuming unsatisfactory"
                }
            
            print(f"[DEBUG] DEBUG - Raw evaluation response (first 200 chars): {response[:200]}")
            
            #clean response - extract JSON even if there's text before it
            import re
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                response = json_match.group(0)
            else:
                # Fallback to old method
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1]
            
            response = response.strip()
            
            if not response:
                print("[DEBUG] DEBUG - Response is empty after cleaning")
                return {
                    "is_satisfactory": False,
                    "confidence": 0.0,
                    "reason": "Empty response after cleaning"
                }
            
            # Fix: Use raw string decoding to handle escaped quotes properly
            try:
                evaluation = json.loads(response)
            except json.JSONDecodeError:
                # Try fixing escaped quotes
                response = response.replace("\\", "\\\\")
                evaluation = json.loads(response)
            print(f"[TEST] Answer evaluation: {evaluation.get('reason','no reason provided')}")
            return evaluation
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parsing error in evaluation: {e}")
            print(f"[DEBUG] DEBUG - Attempted to parse: '{response}'")
            return {
                "is_satisfactory": False,
                "confidence": 0.0,
                "reason": "JSON parsing failed - assuming unsatisfactory"
            }
        except Exception as e:
            print(f"[WARN] Error evaluating answer quality: {e}")
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
            
            # Debug: Show what's in the vector store
            print(f"[DEBUG] DEBUG - Files in vector store: {processed_files}")
            print(f"[DEBUG] DEBUG - Total chunks in DB: {collection.count()}")
            
        except Exception as e:
            print(f"[WARN] Error querying vector store: {e}")
            processed_files = set()
        #find unprocessed files
        unprocessed_files = [f for f in all_files if f.name not in processed_files]
        
        if unprocessed_files:
            print(f"[LIST] Found {len(unprocessed_files)} unprocessed files:")
            for f in unprocessed_files:
                print(f"   - {f.name}")
        else:
            print("[SUCCESS] All files have been processed!")
        return unprocessed_files
    
    def process_new_files(self)->bool:
        """trigger document processing for new files not yet in vector store"""
        try:
            print("[START] Processing new files with Advanced Document Processor...")
            result = subprocess.run(
                ["uv", "run", "advanced_document_processor.py"],
                capture_output=True,
                text=True,
                timeout=200
                
            )
            if result.returncode ==0:
                print(" new files processed successfully.")
                print("[RELOAD] Reloading vector store to pick up new documents...")
                
                # Reload the entire vector store to get new documents
                use_local = self.config.USE_LOCAL_LLM
                if not self.vector_store.initialize(use_local_embeddings=use_local):
                    print("[WARN] Failed to reload vector store")
                    return False
                
                # Reinitialize retriever with more results
                self.retriever = self.vector_store.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                print("[OK] Vector store reloaded successfully")
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
             self.logger.info(f"Performing web search for: {question[:100]}")
             search_results = self.web_search.run(question)
             
             #ask llm to synthesize a brief answer from search results
             synthesis_prompt = f"""Based on these web search results, answer the user's question.

Question: {question}

Search Results:
{search_results}

Provide a clear, concise answer:"""
             answer = self.llm.invoke(synthesis_prompt)
             self.logger.info("Web search completed successfully")
             return f"**Web Search Result:**\n\n{answer} \n\n_source : DuckDuckGo Search_"
         except Exception as e:
             self.logger.error(f"Web search failed: {e}")
             return " Error performing web search."  
                                            
        
    def detect_intent_keyword(self, question: str) -> Dict[str, str]:
        """Fast, free keyword detection to save time/quota"""
        q = question.lower()
        
        # List Files
        if any(k in q for k in ["list files", "list directory", "show files"]) or q.startswith("ls"):
            path = "."
            # Extract path from "ls <path>" or "ls in <path>"
            if q.startswith("ls "):
                path_part = question[3:].strip()
                if path_part and not path_part.startswith("in"):
                    path = path_part.split()[0]
            elif "in " in q:
                parts = q.split("in ")
                if len(parts) > 1: 
                    path = parts[1].strip().split()[0]
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
        
        # Get File Info
        if any(k in q for k in ["file info", "stat ", "info about"]):
            words = question.split()
            filename = words[-1]
            for word in words:
                if "." in word and len(word) > 2: 
                    filename = word
                    break
            return {"action": "get_file_info", "parameter": filename}
        
        # Create Directory
        if q.startswith("mkdir "):
            path = question[6:].strip()
            return {"action": "create_directory", "parameter": path, "requires_permission": True}
        
        # Move/Rename
        if q.startswith("mv ") or q.startswith("move "):
            parts = question.split()
            if len(parts) >= 3:
                return {"action": "move_file", "parameter": f"{parts[1]}|{parts[2]}", "requires_permission": True}
        
        # Delete
        if q.startswith("rm ") or q.startswith("delete "):
            path = question.split()[1] if len(question.split()) > 1 else ""
            return {"action": "delete_file", "parameter": path, "requires_permission": True}
        
        # Create/Write File
        if q.startswith("touch ") or q.startswith("create file "):
            path = question.split()[-1]
            return {"action": "write_file", "parameter": f"{path}|", "requires_permission": True}

        return None 

    def decide_action(self, question: str) -> Dict[str, Any]:
        """Ask LLM to decide which MCP tool to use (only called after RAG fails)"""
        
        # 1. Try fast keyword match first
        keyword_decision = self.detect_intent_keyword(question)
        if keyword_decision:
            print("[FAST] Fast keyword match used")
            return keyword_decision

        # 2. Fallback to LLM Router
        print("[THINK] Asking LLM which filesystem tool to use...")
        router_template = """The user asked a question, but we found nothing in the course documents.
Now we need to use filesystem tools.

Available tools:
READ OPERATIONS:
- list_files: List all files in current directory or a path
- search_files: Find files matching a pattern (e.g., "*.py" for Python files)
- read_file: Read content of a specific file
- file_info: Get file metadata (size, date)

WRITE OPERATIONS (will ask for permission):
- create_directory: Create a new folder
- write_file: Create or update a file (use parameter format: "path|content")
- move_file: Move or rename files (use parameter format: "source|destination")
- delete_file: Delete a file or directory

Question: {question}

Examples:
- "show me python files" → {{"action": "search_files", "parameter": "*.py"}}
- "create a backup folder" → {{"action": "create_directory", "parameter": "backup"}}
- "move test.txt to archive" → {{"action": "move_file", "parameter": "test.txt|archive/test.txt"}}
- "list all files" → {{"action": "list_files", "parameter": "."}}

Return ONLY JSON with "action" and "parameter":"""

        prompt = PromptTemplate(template=router_template, input_variables=["question"])
        formatted_prompt = prompt.format(question=question)
        
        try:
            response_text = self.llm.invoke(formatted_prompt)
            
            print(f"[DEBUG] DEBUG - Raw LLM response: {response_text[:200]}...")
            
            # Clean up response
            if not response_text or response_text.strip() == "":
                print("[WARN] Empty response, defaulting to list_files")
                return {"action": "list_files", "parameter": "."}
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1]
            
            response_text = response_text.strip()
            print(f"[DEBUG] DEBUG - Cleaned response: {response_text}")
            
            decision = json.loads(response_text)
            return decision
            
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parsing failed: {e}, defaulting to list_files")
            return {"action": "list_files", "parameter": "."}
        except Exception as e:
            print(f"[WARN] Router error: {e}, defaulting to list_files")
            return {"action": "list_files", "parameter": "."}        
        
    
        
    def create_rag_prompt(self):
        """crreate prompt template for RAG"""
        template = """
Tu es un assistant intelligent qui aide les étudiants avec leurs documents de cours.

RÈGLES IMPORTANTES:
1. Réponds UNIQUEMENT avec les informations du contexte ci-dessous
2. Ne fais AUCUNE supposition ou invention
3. Si l'information n'est PAS dans le contexte, dis "Je ne trouve pas cette information dans les documents"
4. Cite les passages exacts du contexte quand possible
5. Si tu trouves une définition ou un acronyme, copie-le EXACTEMENT comme écrit

Contexte:
{context}

Question: {question}

Réponse (basée UNIQUEMENT sur le contexte ci-dessus):
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    

    def query(self, question: str) -> Dict[str, Any]:
        """Main query handler with multi-stage intelligence"""
        query_start_time = time.time()
        
        if not self.vector_store or not self.llm:
            self.logger.error("Query attempted with uninitialized RAG system")
            return {
                "answer": "[ERROR] RAG system not initialized.",
                "sources": []
            }
        
        # Skip caching for commands
        if question.lower() in ['quit', 'exit', 'q', 'ls', 'list', 'cache stats', 'cache clear']:
            return {"answer": "Command recognized but not processed yet", "sources": [], "context": "command"}
        
        self.logger.debug(f"Processing query: {question[:100]}")
        
        # CACHE CHECK: Try to get cached result first
        cached_result = self.query_cache.get(question, k=10)
        if cached_result:
            duration = time.time() - query_start_time
            self.logger.log_query(question, "cache", duration, success=True)
            self.logger.info("Cache hit - returning cached result")
            return cached_result
        
        self.logger.debug("Cache miss - processing query...")
        
        # Fast path: Check for simple MCP commands first
        keyword_decision = self.detect_intent_keyword(question)
        if keyword_decision:
            print("[FAST] Fast keyword match - executing MCP tool directly")
            action = keyword_decision['action']
            parameter = keyword_decision['parameter']
            
            # Check if permission is required
            if keyword_decision.get('requires_permission', False):
                print(f"\n[WARN]  PERMISSION REQUIRED:")
                print(f"   Action: {action}")
                print(f"   Target: {parameter}")
                confirm = input("\n   Proceed? (yes/no): ").strip().lower()
                if confirm not in ['yes', 'y']:
                    print("[ERROR] Operation cancelled by user.")
                    return {
                        "answer": "[ERROR] Operation cancelled by user.",
                        "sources": [],
                        "context": "Cancelled"
                    }
            
            tool_output = self.execute_tool_action(action, parameter)
            return {
                "answer": f"**Tool Output ({action}):**\n\n{tool_output}",
                "sources": [],
                "context": "MCP Tool (keyword detection)"
            }
        
        # STAGE 1: Try RAG
        print("[SEARCH] Stage 1: Searching course documents...")
        try:
            # Reformulate question for better semantic search
            reformulation_prompt = f"""Reformule cette question en termes techniques pour une recherche dans des documents académiques.
Garde seulement les mots-clés importants et les termes techniques.

Question: {question}

Reformulation (mots-clés uniquement, pas de phrase complète):"""
            
            try:
                keywords = self.llm.invoke(reformulation_prompt).strip()
                print(f"[SEARCH] Keywords extracted: {keywords[:100]}")
                # Search with both original question and keywords
                docs1 = self.vector_store.search(question, k=5)
                docs2 = self.vector_store.search(keywords, k=5)
                # Combine and deduplicate
                seen_content = set()
                relevant_docs = []
                for doc in docs1 + docs2:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        relevant_docs.append(doc)
                        if len(relevant_docs) >= 10:
                            break
            except:
                # Fallback to simple search
                relevant_docs = self.retriever.invoke(question)
            
            print(f"[DOC] Retrieved {len(relevant_docs)} documents")
            
            # DEBUG: Show what was retrieved
            for i, doc in enumerate(relevant_docs[:3]):  # Show first 3
                print(f"[DEBUG] Chunk {i+1}: {doc.page_content[:100]}...")
                print(f"   From: {doc.metadata.get('file_name', 'Unknown')}")
            
            if relevant_docs and len(relevant_docs) > 0:
                context = "\n\n".join([d.page_content for d in relevant_docs])
                prompt_template = self.create_rag_prompt()
                prompt = prompt_template.format(context=context, question=question)
                
                print("[AI] Generating answer from documents...")
                rag_answer = self.llm.invoke(prompt)
                
                print(f"[DEBUG] DEBUG - RAG Answer preview: {rag_answer[:150]}...")
                
                sources = [
                    {
                        "file_name": doc.metadata.get("file_name", "Unknown"),
                        "chunk_index": doc.metadata.get("chunk_index", 0)
                    }
                    for doc in relevant_docs
                ]
                
                # STAGE 2: Evaluate answer quality
                print("[TEST] Stage 2: Evaluating answer quality...")
                evaluation = self.evaluate_answer_quality(question, rag_answer, sources)
                
                print(f"[DEBUG] DEBUG - Evaluation result: {evaluation}")
                
                if evaluation.get("is_satisfactory", False):
                    duration = time.time() - query_start_time
                    self.logger.log_query(question, "rag", duration, success=True)
                    self.logger.info(f"RAG answer satisfactory (confidence: {evaluation.get('confidence', 0.0)})")
                    final_result = {
                        "answer": rag_answer,
                        "sources": sources,
                        "context": "RAG",
                        "confidence": evaluation.get("confidence", 0.0)
                    }
                    # Cache the successful RAG result
                    self.query_cache.set(question, final_result, k=10)
                    return final_result
                else:
                    self.logger.warning(f"RAG answer not satisfactory: {evaluation.get('reason')}")
        
        except Exception as e:
            print(f"[WARN] RAG failed: {e}")
        
        # STAGE 3: Ask LLM to check for unprocessed files using MCP
        print("[THINK] Stage 3: Asking LLM to check for unprocessed files...")
        
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
            print(f"[TOOL] LLM decided: {decision.get('action')} - {decision.get('reasoning', 'No reason')}")
            
            # Check if this action requires permission
            action = decision.get("action")
            parameter = decision.get("parameter")
            
            # Determine if permission is needed
            write_actions = ["create_directory", "write_file", "move_file", "delete_file"]
            if action in write_actions:
                print(f"\n  PERMISSION REQUIRED:")
                print(f"   Action: {action}")
                print(f"   Target: {parameter}")
                confirm = input("\n   Proceed? (yes/no): ").strip().lower()
                if confirm not in ['yes', 'y']:
                    print(" Operation cancelled by user.")
                    return "Operation cancelled."
            
            # Execute the MCP tool the LLM chose
            tool_result = self.execute_tool_action(action, parameter)
            
            print(f"[FOLDER] MCP Tool Result:\n{tool_result[:200]}...")
            
            # STAGE 4: Check for unprocessed files directly (not relying on LLM guessing)
            print("[LIST] Stage 4: Checking for unprocessed files...")
            unprocessed = self.find_unprocessed_files()
            
            if unprocessed and len(unprocessed) > 0:
                print(f"[RELOAD] Found {len(unprocessed)} unprocessed files. Processing...")
                
                if self.process_new_files():
                    print("[RETRY] Retrying RAG with newly processed documents...")
                    
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
                        
                        final_result = {
                            "answer": rag_answer,
                            "sources": sources,
                            "context": "RAG (after processing new files)"
                        }
                        # Cache the result
                        self.query_cache.set(question, final_result, k=10)
                        return final_result
            else:
                print("[LIST] No new files to process.")
                
                final_decision_prompt = f"""The filesystem check showed:
{tool_result}

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
                print(f"[TARGET] Final decision: {final_decision.get('action')} - {final_decision.get('reasoning')}")
                
                if final_decision.get("action") == "stop":
                    return {
                        "answer": f"[ERROR] Unable to answer this question with available resources.\n\n{final_decision.get('reasoning')}",
                        "sources": [],
                        "context": "No suitable data"
                    }
                # If action is "web_search", continue to Stage 5 below
        
        except Exception as e:
            print(f"[WARN] File check failed: {e}")
        
        # STAGE 4.5: Keyword search fallback before web search
        print("[SEARCH] Stage 4.5: Trying keyword search in all document chunks...")
        
        # Extract keywords - prioritize quoted terms
        import re
        quoted_terms = re.findall(r'"([^"]+)"', question)
        
        keywords_to_search = []
        if quoted_terms:
            keywords_to_search = quoted_terms
            print(f"[FIND] Found quoted terms: {quoted_terms}")
        else:
            # Fallback to significant words
            keywords = question.split()
            for word in keywords:
                if len(word) > 2 and word.lower() not in ['que', 'est', 'le', 'la', 'les', 'un', 'une', 'des', 'what', 'is', 'the', 'selon', 'dans', 'expliquer']:
                    keywords_to_search.append(word.strip('?.,;:'))
                    break
        
        if keywords_to_search:
            # If we have multiple quoted terms, search for chunks containing ALL of them
            if len(keywords_to_search) > 1:
                print(f"[FIND] Searching for chunks containing ALL terms: {keywords_to_search}...")
                try:
                    all_docs = self.vector_store.vector_store._collection.get(
                        include=["documents", "metadatas"]
                    )
                    
                    matching_chunks = []
                    
                    for i, doc_text in enumerate(all_docs['documents']):
                        # Check if ALL keywords are in this chunk
                        all_found = True
                        for term in keywords_to_search:
                            # For short terms, use word boundary
                            if len(term) <= 3:
                                pattern = r'\b' + re.escape(term) + r'\b'
                                if not re.search(pattern, doc_text, re.IGNORECASE):
                                    all_found = False
                                    break
                            else:
                                if term.lower() not in doc_text.lower():
                                    all_found = False
                                    break
                        
                        if all_found:
                            matching_chunks.append({
                                'text': doc_text,
                                'metadata': all_docs['metadatas'][i]
                            })
                            if len(matching_chunks) >= 5:
                                break
                    
                    if matching_chunks:
                        print(f"[SUCCESS] Found {len(matching_chunks)} chunks containing all terms")
                        context = "\n\n".join([chunk['text'] for chunk in matching_chunks])
                        
                        prompt_template = self.create_rag_prompt()
                        prompt = prompt_template.format(context=context, question=question)
                        
                        keyword_answer = self.llm.invoke(prompt)
                        
                        final_result = {
                            "answer": keyword_answer,
                            "sources": [{'file_name': chunk['metadata'].get('file_name', 'Unknown')} for chunk in matching_chunks],
                            "context": "Keyword Search (multi-term)"
                        }
                        # Cache keyword search results
                        self.query_cache.set(question, final_result, k=10)
                        return final_result
                except Exception as e:
                    print(f"[WARN] Multi-term search failed: {e}")
            
            # Fallback: Try each keyword individually
            for main_keyword in keywords_to_search[:3]:  # Limit to first 3 keywords
                print(f"[FIND] Searching for keyword: '{main_keyword}'...")
                try:
                    all_docs = self.vector_store.vector_store._collection.get(
                        include=["documents", "metadatas"]
                    )
                    
                    matching_chunks = []
                    search_term = main_keyword.lower()
                    
                    # For single letters or short terms, use word boundary search
                    if len(main_keyword) <= 3:
                        pattern = r'\b' + re.escape(main_keyword) + r'\b'
                        for i, doc_text in enumerate(all_docs['documents']):
                            if re.search(pattern, doc_text, re.IGNORECASE):
                                matching_chunks.append({
                                    'text': doc_text,
                                    'metadata': all_docs['metadatas'][i]
                                })
                                if len(matching_chunks) >= 5:
                                    break
                    else:
                        for i, doc_text in enumerate(all_docs['documents']):
                            if search_term in doc_text.lower():
                                matching_chunks.append({
                                    'text': doc_text,
                                    'metadata': all_docs['metadatas'][i]
                                })
                                if len(matching_chunks) >= 5:
                                    break
                    
                    if matching_chunks:
                        print(f"[SUCCESS] Found {len(matching_chunks)} chunks with '{main_keyword}'")
                        context = "\n\n".join([chunk['text'] for chunk in matching_chunks])
                        
                        prompt_template = self.create_rag_prompt()
                        prompt = prompt_template.format(context=context, question=question)
                        
                        keyword_answer = self.llm.invoke(prompt)
                        
                        final_result = {
                            "answer": keyword_answer,
                            "sources": [{'file_name': chunk['metadata'].get('file_name', 'Unknown')} for chunk in matching_chunks],
                            "context": "Keyword Search"
                        }
                        # Cache keyword search results
                        self.query_cache.set(question, final_result, k=10)
                        return final_result
                except Exception as e:
                    print(f"[WARN] Keyword search failed for '{main_keyword}': {e}")
                    continue
        
        # STAGE 5: Web search fallback (only if we get here)
        self.logger.info("Stage 5: Falling back to web search")
        web_answer = self.web_search_fallback(question)
        
        duration = time.time() - query_start_time
        self.logger.log_query(question, "web", duration, success=True)
        
        final_result = {
            "answer": web_answer,
            "sources": [],
            "context": "Web Search"
        }
        # Cache web search results
        self.query_cache.set(question, final_result, k=10)
        return final_result



               
       
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
                    
        
        
        
def main():
    """main rag system interface """
    rag= RAGSystem()
    
    # Initialize the system
    if not rag.initialize():
        rag_logger.error("RAG system initialization failed")
        print("[ERROR] RAG system initialization failed.")
        return
    
    if not rag.vector_store or not rag.llm:
        rag_logger.error("RAG system initialization failed - missing components")
        print("[ERROR] RAG system initialization failed.")
        return
    
    print("\n" + "="*60)
    print(" RAG System with MCP Protocol Ready!")
    print("="*60)
    print("\nAvailable commands:")
    print("  - Ask questions about your documents")
    print("  - 'ls' or 'list' - list files via MCP")
    print("  - 'search <pattern>' - search files via MCP (e.g., 'search *.pdf')")
    print("  - 'read <file>' - read file via MCP")
    print("  - 'cache stats' - show cache statistics")
    print("  - 'cache clear' - clear query cache")
    print("  - 'exit' or 'quit' - exit the system")
    print("="*60 + "\n")
    
    try:
        while True:
            question = input(" Your question or command: ").strip()
            if not question:
                continue
            if question.lower() in ['exit', 'quit', 'q']:
                print(" Goodbye!")
                break
            
            # Cache management commands
            if question.lower() == 'cache stats':
                stats = rag.query_cache.get_stats()
                print(f"\n{'='*60}")
                print(" Cache Statistics")
                print(f"{'='*60}")
                print(f"Total Queries: {stats['total_queries']}")
                print(f"Cache Hits: {stats['hits']}")
                print(f"Cache Misses: {stats['misses']}")
                print(f"Hit Rate: {stats['hit_rate']}")
                print(f"Cached Entries: {stats['cached_entries']}")
                print(f"Evictions: {stats['evictions']}")
                print(f"TTL: {stats['ttl_minutes']} minutes")
                print(f"{'='*60}\n")
                continue
            
            if question.lower() == 'cache clear':
                rag.query_cache.clear()
                print("\n[OK] Cache cleared successfully!\n")
                continue
            
            result = rag.query(question)
            
            print(f"\n{'='*60}")
            print(" **Response:**")
            print(f"{'='*60}")
            print(result['answer'])
            print(f"{'='*60}\n")
    finally:
        #cleanup mcp connection
        if rag:
            rag.close()
            # Export final stats
            rag_logger.export_stats()
            rag_logger.info("System closed successfully")
            print("[SUCCESS] System closed successfully")           

if __name__ == "__main__":
    main()