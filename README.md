# 🎓 Smart Notes Assistant (RAG + MCP)

An intelligent study companion that combines **Retrieval-Augmented Generation (RAG)** with **Model Context Protocol (MCP)** tools. This assistant helps you interact with your course documents (PDFs) and perform local filesystem operations directly from a chat interface.

## 🚀 Features

*   **RAG System**: Uses **Google Gemini 2.0 Flash** and **ChromaDB** to answer questions based on your local PDF documents.
*   **MCP Tools**: Integrated tools to interact with your computer:
    *   `list files` / `ls`: Browse directories.
    *   `read [file]` / `cat`: Read file contents.
    *   `search [term]`: Find files by name.
    *   `file info`: Get file metadata.
*   **Smart Intent Detection**: Automatically detects if you need a local tool or an AI answer to save API quota.

## 🛠️ Installation

This project uses `uv` for fast Python package management.

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd RAG_MCP_PROJECT
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 📖 Usage

1.  **Place your documents**: Put your PDF or DOCX course files in the `documents/` folder.

2.  **Run the assistant**:
    ```bash
    uv run simple_rag.py
    ```

3.  **Interact**:
    *   Ask questions: *"What is quality control?"*
    *   Use tools: *"list files in documents"*, *"read config.py"*

## 🏗️ Project Structure

*   `simple_rag.py`: Main application entry point and RAG logic.
*   `mcp_server.py`: Real MCP server used at runtime (stdio, `list_tools`/`call_tool`).
*   `mcp_tools.py`: Legacy helper module (not in the active MCP runtime path).
*   `vector_store.py`: Manages ChromaDB vector database.
*   `document_processor.py`: Handles PDF/DOCX loading and chunking.
*   `config.py`: Configuration settings.

### MCP Runtime Note

The active MCP flow is:

1. `simple_rag.py` starts `mcp_server.py` via stdio.
2. `simple_rag.py` calls MCP tools through `mcp_session.call_tool(...)`.
3. `mcp_server.py` executes the registered tool handlers.

`mcp_tools.py` is not used by the runtime MCP server.

## 🤖 Technologies

*   Python 3.12+
*   LangChain
*   Google Generative AI (Gemini)
*   ChromaDB
*   UV (Package Manager)
