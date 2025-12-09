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
*   `mcp_tools.py`: Implementation of filesystem tools.
*   `vector_store.py`: Manages ChromaDB vector database.
*   `document_processor.py`: Handles PDF/DOCX loading and chunking.
*   `config.py`: Configuration settings.

## 🤖 Technologies

*   Python 3.12+
*   LangChain
*   Google Generative AI (Gemini)
*   ChromaDB
*   UV (Package Manager)
