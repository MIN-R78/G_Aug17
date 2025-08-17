## PDF RAG Toolkit

A comprehensive toolkit for parsing PDF documents, generating text embeddings, and running RAG (Retrieval-Augmented Generation) pipelines. Built with MCP (Model Context Protocol) server architecture for flexible tool integration.

## Core Features

- **PDF Parser**: Dual support for PyPDF2 and pdfplumber with intelligent text chunking
- **Vector Embeddings**: Generate semantic vector representations using sentence-transformers
- **Vector Search**: Fast similarity search and retrieval powered by FAISS
- **MCP Server**: Standardized tool interface supporting tool registration and execution
- **Batch Processing**: Efficient handling of multiple PDF files with folder indexing

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/MIN-R78/G_Aug17.git
cd G_Aug17
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tests
```bash
python test_rag_mcp.py
```

### 4. Start Server
```bash
python rag_mcp_server.py
```

## Main Components

- **`pdf_parser_mcp.py`**: PDF text extraction tool with smart chunking
- **`embedding_mcp.py`**: Text vectorization tool with similarity search
- **`rag_mcp_server.py`**: Main MCP server coordinating all tools
- **`test_rag_mcp.py`**: Comprehensive test suite for all functionality

## Usage Examples

```python
# PDF Parsing
from pdf_parser_mcp import PDFParserTool
parser = PDFParserTool()
result = parser.execute({
    "operation": "parse_pdf",
    "pdf_path": "document.pdf",
    "parser_type": "advanced",
    "chunk_size": 3
})

# Text Vectorization
from embedding_mcp import EmbeddingTool
embedder = EmbeddingTool()
result = embedder.execute({
    "operation": "embed",
    "texts": ["Hello world", "Another text"],
    "model_name": "all-MiniLM-L6-v2"
})
```

## Requirements

- Python 3.8+
- Core dependencies: PyPDF2, pdfplumber, sentence-transformers, faiss-cpu
- See `requirements.txt` for complete dependency list

## Project Highlights

- **Modular Design**: Each tool is an independent class, easy to extend and maintain
- **Error Handling**: Comprehensive input validation and exception handling
- **Performance Optimization**: Batch processing and memory management support
- **Standard Interface**: MCP protocol compliance for easy system integration
