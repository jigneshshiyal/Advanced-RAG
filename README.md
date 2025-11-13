RAG (PDF -> Embeddings -> Query) Backend
=====================================

Summary
-------
This repository contains a small Retrieval-Augmented Generation (RAG) backend that:

- Extracts text from PDF files
- Splits text into chunks
- Computes embeddings and stores them in a local Chroma vector store
- Exposes a FastAPI service with endpoints to upload PDFs (embed) and query the data

The FastAPI app lives in `backend/main.py`. The core components are in `backend/`:

- `pdf_utils.py` — PDF text extraction and cleaning
- `chunking.py` — splitting documents into chunks and generating stable chunk IDs
- `embedding_model.py` — embedding wrapper (Gemini in this project)
- `embed_store.py` — batch embedding + store interface
- `vectordb.py` — chroma client / collection config
- `retrievel.py` — retrieval, reranking, and answer generation using the LLM
- `llm.py` — LLM wrapper

Requirements
------------
Python 3.10+ recommended. The project uses several third-party packages (examples below). Create a virtual environment before installing.

Minimal recommended packages (example):

- fastapi
- uvicorn[standard]
- chromadb
- langchain-core (or langchain packages used)
- langchain-google-genai (for Gemini embeddings)
- sentence-transformers
- rank_bm25
- pymupdf
- beautifulsoup4
- numpy
- scipy
- cross-encoder
- python-dotenv

You can create a `requirements.txt` with the packages you need. Example (not exhaustive):

```text
fastapi
uvicorn[standard]
chromadb
langchain-core
langchain-google-genai
sentence-transformers
rank_bm25
pymupdf
beautifulsoup4
numpy
scipy
python-dotenv
cross-encoder
```

Setup
-----
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install packages:

```powershell
pip install -r requirements.txt
```

3. Set environment variables. The project expects at least:

- GEMINI_API_KEY — your Google/Gemini API key
- GEMINI_MODEL — optional, defaults to `models/gemini-2.5-flash-lite`

Create a `.env` file in the project root or set OS environment variables. Example `.env`:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-2.5-flash-lite
```

Usage
-----
Start the API (from project root):

```powershell
# Recommended: run from project root so the package imports work
uvicorn backend.main:app --reload --port 8000
```

Or run the script directly (development quick-start):

```powershell
python backend/main.py
```

API Endpoints
-------------
1) POST /embed_pdf

- Accepts multipart form uploads (one or more PDF files) using field name `files`.
- The server will:
  - Save uploaded PDFs temporarily to `backend/uploaded_pdfs/`
  - Extract pages and clean text (`pdf_utils.pdfs_loader`)
  - Chunk pages (`chunking.chunking_doc`)
  - Compute embeddings and add them to the chroma collection (`embed_store.embed_and_store`)
- Response example:

```json
{ "status": "ok", "uploaded_files": 1, "chunks": 12 }
```

PowerShell curl example (requires curl or use Invoke-RestMethod / Postman):

```powershell
curl -F "files=@C:\path\to\doc.pdf" http://127.0.0.1:8000/embed_pdf
```

2) POST /query

- Accepts JSON body: `{ "query": "Your question", "k": 5 }` (k is accepted but the backend's retriever currently uses internal defaults).
- Returns the generated answer and the documents used (content and metadata).

PowerShell example:

```powershell
curl -H "Content-Type: application/json" -d '{"query":"What does the PDF say about X?"}' http://127.0.0.1:8000/query
```

Notes and Troubleshooting
-------------------------
- Relative import issues: Some modules used relative imports (for example `from .config import GEMINI_API_KEY`). `backend/main.py` inserts the backend directory into `sys.path` so running `uvicorn backend.main:app` from the project root will avoid the "attempted relative import with no known parent package" error. If you run `python backend/main.py` directly from inside the `backend/` directory, you may encounter import errors — prefer running `uvicorn backend.main:app` from the project root.

- Chroma DB: The chroma persistent store is configured in `backend/vectordb.py` and by default stores data under `backend/store_emb`. Ensure that folder is writable.

- Large PDFs: Embedding is batched; the embedding provider may rate-limit or throttle. Check `embedding_model.gemini_embedding_batch` for batch size and sleep settings.

- LLM and embeddings: This project uses Google Gemini via `langchain-google-genai` packages. Ensure your API key has the required access and quota.

- Background processing: Currently, the `/embed_pdf` endpoint processes embeddings synchronously. For large or many files, consider offloading to a background worker or queue and returning an upload job id.

Next steps (suggested)
----------------------
- Add `requirements.txt` (if you want I can generate a minimal one based on imports).
- Make embedding asynchronous / background job.
- Add tests for endpoints using FastAPI TestClient.
- Secure the endpoints with API keys or OAuth for production usage.

Contact / Help
---------------
If you want, I can:
- Generate a `requirements.txt` pinned to specific versions
- Add unit/integration tests
- Wire `k` through to the retriever and add query parameters to the API

Happy to make any of those improvements — tell me which one you want next.