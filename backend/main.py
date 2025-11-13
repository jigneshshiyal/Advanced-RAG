from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys
import shutil
import uuid

# Ensure local backend modules are importable when running this script directly
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
	sys.path.insert(0, BASE_DIR)

import pdf_utils
import chunking
import embed_store
import retrievel

app = FastAPI(title="RAG Backend API")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
	query: str
	k: int = 5


@app.post("/embed_pdf")
def embed_pdf(files: List[UploadFile] = File(...)):
	"""
	Upload one or more PDF files. The server will save them temporarily, extract text,
	chunk the text, create embeddings and store them in the vector DB.
	Returns a small report about how many chunks were created and queued for embedding.
	"""
	if not files:
		raise HTTPException(status_code=400, detail="No files uploaded")

	saved_paths = []
	try:
		for f in files:
			filename = f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}"
			dest = os.path.join(UPLOAD_DIR, filename)
			with open(dest, "wb") as out_f:
				shutil.copyfileobj(f.file, out_f)
			saved_paths.append(dest)

		# extract pages -> list of {'text', 'metadata'} per page
		docs = pdf_utils.pdfs_loader(saved_paths)

		# chunk the docs (returns items with 'text' and 'metadata' and 'id')
		chunks = chunking.chunking_doc(docs)

		# embed and store
		embed_store.embed_and_store(chunks)

		return {"status": "ok", "uploaded_files": len(saved_paths), "chunks": len(chunks)}

	finally:
		# keep uploaded files for debugging; optionally remove them here if you want
		for p in saved_paths:
			try:
				os.remove(p)
			except Exception:
				pass


@app.post("/query")
def query(request: QueryRequest):
	"""
	Query the RAG system. Returns generated answer and the list of documents used
	(content + metadata) for traceability.
	"""
	if not request.query or request.query.strip() == "":
		raise HTTPException(status_code=400, detail="query must be provided")

	result = retrievel.answer_query(request.query)

	# Optionally, trim large document contents in the API response depending on needs
	return {"query": request.query, "answer": result.get("answer"), "documents": result.get("documents")}


if __name__ == "__main__":
	# Run with: python backend/main.py or use uvicorn from project root: uvicorn backend.main:app --reload
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

