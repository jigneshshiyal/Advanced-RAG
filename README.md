Advanced-RAG: Production-Ready Retrieval-Augmented Generation
==============================================================

**Summary**

This repository contains a production-grade **Retrieval-Augmented Generation (RAG)** system that combines advanced PDF processing, multi-strategy retrieval, intelligent re-ranking, and LLM integration. It includes:

- **PDF Processing**: Extracts, cleans, and intelligently chunks PDF documents
- **Hybrid Retrieval**: Combines semantic similarity, MMR diversity, and BM25 lexical matching
- **Cross-Encoder Re-ranking**: Uses a fine-tuned cross-encoder model for precise relevance scoring
- **Vector Storage**: Local Chroma vector database for efficient similarity search
- **FastAPI Backend**: RESTful API for document ingestion and querying
- **Streamlit UI**: User-friendly interface for uploading PDFs and asking questions
- **Full Traceability**: Returns retrieved documents with detailed scoring metadata

**Key Features**

- 🔄 **Hybrid Retrieval System**: Intelligently combines three retrieval strategies (semantic + MMR + BM25) with normalized scoring
- 🎯 **Cross-Encoder Re-ranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise document ranking
- 📄 **Intelligent Chunking**: Stable chunk IDs with metadata preservation
- 🧠 **Gemini Integration**: Leverages Google Gemini 2.5 Flash for embeddings and answer generation
- 🔍 **Citation Tracking**: Includes source metadata and relevance scores in responses
- 🌐 **CORS-Enabled API**: Ready for frontend integration
- 💾 **Local Vector Database**: Chroma with persistent storage

**Project Structure**

The FastAPI app lives in `backend/main.py`. Core components:

| File | Purpose |
|------|---------|
| `pdf_utils.py` | PDF text extraction and cleaning |
| `chunking.py` | Document chunking with stable IDs |
| `embedding_model.py` | Embedding wrapper (Gemini) |
| `embed_store.py` | Batch embedding and vector storage |
| `vectordb.py` | Chroma client and collection configuration |
| `retrievel.py` | Hybrid retrieval, re-ranking, and LLM answer generation |
| `llm.py` | LLM wrapper for Gemini models |
| `config.py` | Configuration and environment setup |
| `streamlit_app.py` | Interactive web UI for document upload and querying |

Requirements
------------

- **Python 3.10+** recommended
- Create a virtual environment before installing dependencies

**Core Dependencies:**

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for REST API |
| `uvicorn[standard]` | ASGI server |
| `chromadb` | Vector database for embeddings |
| `langchain-core` | Core LangChain abstractions |
| `langchain-google-genai` | Google Gemini integration |
| `sentence-transformers` | Embedding and cross-encoder models |
| `rank_bm25` | BM25 lexical ranking algorithm |
| `pymupdf` | PDF text extraction |
| `beautifulsoup4` | HTML/text parsing and cleaning |
| `numpy` | Numerical computations |
| `scipy` | Scientific computing (softmax, normalization) |
| `cross-encoder` | Cross-encoder model for re-ranking |
| `python-dotenv` | Environment variable management |
| `streamlit` | Web UI framework |

**Create `requirements.txt`:**

```text
fastapi
uvicorn[standard]
chromadb
langchain-core
langchain-google-genai
langchain-chroma
sentence-transformers
rank_bm25
pymupdf
beautifulsoup4
numpy
scipy
python-dotenv
cross-encoder
streamlit
requests
```

Setup
-----

1. **Create and activate a virtual environment:**

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Windows (cmd):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   The project requires the following environment variables:

   - `GEMINI_API_KEY` — Your Google Gemini API key (required)
   - `GEMINI_MODEL` — Optional, defaults to `models/gemini-2.5-flash-lite`
   - `API_URL` — Optional, used by Streamlit UI for API endpoint (defaults to `http://127.0.0.1:8000`)

   **Create a `.env` file in the project root:**

   ```ini
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=models/gemini-2.5-flash-lite
   API_URL=http://127.0.0.1:8000
   ```

   Alternatively, set these as OS environment variables.

Usage
-----

**Start the FastAPI Backend (from project root):**

```cmd
uvicorn backend.main:app --reload --port 8000
```

Or run the script directly (development quick-start):

```cmd
python backend/main.py
```

**Start the Streamlit UI (from project root, in a separate terminal):**

```cmd
streamlit run streamlit_app.py
```

The Streamlit app will open at `http://localhost:8501` and communicate with the FastAPI backend at `http://127.0.0.1:8000`.

API Endpoints
-------------

### 1) POST /embed_pdf

Upload one or more PDF files to extract, chunk, embed, and store in the vector database.

**Request:**
- Multipart form upload with field name `files`
- Supports multiple PDF files

**Processing Pipeline:**
1. Save uploaded PDFs temporarily to `backend/uploaded_pdfs/`
2. Extract pages and clean text using `pdf_utils.pdfs_loader()`
3. Split pages into chunks using `chunking.chunking_doc()`
4. Compute embeddings and store in Chroma using `embed_store.embed_and_store()`

**Response Example:**
```json
{
  "status": "ok",
  "uploaded_files": 1,
  "chunks": 12
}
```

**Example (PowerShell/curl):**
```powershell
curl -F "files=@C:\path\to\doc.pdf" http://127.0.0.1:8000/embed_pdf
```

**Example (Python):**
```python
import requests

with open("example.pdf", "rb") as f:
    files = {"files": f}
    response = requests.post("http://127.0.0.1:8000/embed_pdf", files=files)
    print(response.json())
```

### 2) POST /query

Query the RAG system to retrieve relevant documents and generate an answer.

**Request Body:**
```json
{
  "query": "Your question here",
  "k": 5
}
```

- `query` (string, required): The question to answer
- `k` (integer, optional): Maximum number of documents to retrieve (default: 5)

**Response:**
Returns the generated answer and a list of retrieved documents with relevance metadata.

```json
{
  "query": "What is RAG?",
  "answer": "Retrieval-Augmented Generation (RAG) is...",
  "documents": [
    {
      "content": "Document text excerpt...",
      "metadata": {
        "source": "document_name.pdf",
        "id": "chunk_123",
        "hybrid_score": 0.85,
        "cross_score": 0.92,
        "final_score": 0.88
      }
    }
  ]
}
```

**Example (Python):**
```python
import requests
import json

payload = {
    "query": "What does the document say?",
    "k": 5
}

response = requests.post(
    "http://127.0.0.1:8000/query",
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"}
)

result = response.json()
print("Answer:", result["answer"])
for doc in result["documents"]:
    print("Source:", doc["metadata"]["source"])
```

**Example (PowerShell/curl):**
```powershell
curl -H "Content-Type: application/json" `
  -d '{"query":"What does the PDF say about X?","k":5}' `
  http://127.0.0.1:8000/query
```

Retrieval Architecture
----------------------

### Hybrid Retrieval Strategy

The system uses a three-pronged retrieval approach (implemented in `retrievel.py`):

1. **Semantic Similarity** (60% weight, configurable):
   - Computes embeddings for the query
   - Measures cosine distance to all document embeddings
   - Normalized score in [0, 1]

2. **Maximal Marginal Relevance (MMR)** (30% weight, configurable):
   - Balances relevance and diversity
   - Reduces redundant results
   - Uses `lambda_mult=0.5` by default

3. **BM25 Lexical Matching** (10% weight, configurable):
   - Traditional full-text search algorithm
   - Captures keyword relevance
   - Normalized alongside semantic scores

Each strategy produces a normalized score, which are then combined into a single **hybrid score** per document.

### Cross-Encoder Re-ranking

After hybrid retrieval, the top candidate documents are re-ranked using:

- **Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Provides more accurate relevance judgments than siamese-style encoders
- Combined with hybrid scores using weighted average (70% cross-encoder, 30% hybrid by default)
- Final score used to select top-k documents for the LLM

### LLM Answer Generation

The selected documents are formatted as context and passed to Google Gemini 2.5 Flash:

- Uses a prompt template to instruct the LLM to cite sources
- Includes document metadata (source, chunk ID, relevance scores)
- Returns a natural language answer with traceability

Performance & Scalability
--------------------------

- **Vector DB**: Chroma local storage with persistent SQLite backend
- **Embedding Model**: Google Gemini 2.5 Flash (fast, low-latency)
- **Cross-Encoder**: Lightweight, typically processes in <100ms per query
- **Chunk Size**: Configurable (default ~300-500 tokens per chunk for balanced context)
- **Batch Processing**: Supports multi-file uploads with efficient batching

Example Workflows
-----------------

**Workflow 1: Upload and Query**

1. Start FastAPI:
   ```cmd
   uvicorn backend.main:app --reload --port 8000
   ```

2. Upload a PDF:
   ```powershell
   curl -F "files=@mydocument.pdf" http://127.0.0.1:8000/embed_pdf
   ```

3. Query the system:
   ```powershell
   curl -H "Content-Type: application/json" `
     -d '{"query":"summarize the main points"}' `
     http://127.0.0.1:8000/query
   ```

**Workflow 2: Using Streamlit UI**

1. Start FastAPI (Terminal 1):
   ```cmd
   uvicorn backend.main:app --reload --port 8000
   ```

2. Start Streamlit (Terminal 2):
   ```cmd
   streamlit run streamlit_app.py
   ```

3. Open browser to `http://localhost:8501` and use the UI to upload and query