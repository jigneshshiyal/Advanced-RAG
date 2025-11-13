import os
import streamlit as st
import requests
import json
from typing import List

# The API base URL. Change if your FastAPI is hosted elsewhere.
# Use Streamlit secrets if available, otherwise fall back to environment variable, then default.
try:
    # Accessing st.secrets may raise if no secrets file exists; guard it.
    api_from_secrets = None
    try:
        api_from_secrets = st.secrets.get("API_URL")
    except Exception:
        api_from_secrets = None
    API_URL = api_from_secrets or os.environ.get("API_URL", "http://127.0.0.1:8000")
except Exception:
    # As a last resort (extremely unlikely), use the default URL
    API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG PDF UI", layout="wide")
st.title("RAG PDF Uploader & Query")
st.write("Upload PDFs to embed them and ask questions over the indexed content.")

tab1, tab2 = st.tabs(["Upload PDFs", "Query"])

with tab1:
    st.header("Upload PDF(s) and create embeddings")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        remove_after = st.checkbox("Remove uploaded file from server after processing", value=True)
    with col2:
        notes = st.text_area("Notes (optional) - stored as metadata","", height=80)

    if st.button("Upload & Embed"):
        if not uploaded_files:
            st.warning("Please select one or more PDF files to upload.")
        else:
            files_payload = []
            for f in uploaded_files:
                # f is a UploadedFile; use f.getvalue() to read bytes
                files_payload.append(("files", (f.name, f.getvalue(), "application/pdf")))

            with st.spinner("Uploading and embedding (this may take a while)..."):
                try:
                    resp = requests.post(f"{API_URL}/embed_pdf", files=files_payload)
                    resp.raise_for_status()
                    data = resp.json()
                    st.success("Upload complete")
                    st.json(data)
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")

with tab2:
    st.header("Query the indexed documents")
    q = st.text_area("Enter your question", height=120)
    k = st.number_input("Number of docs to retrieve (k)", min_value=1, max_value=20, value=5)

    if st.button("Run Query"):
        if not q or q.strip() == "":
            st.warning("Please enter a question.")
        else:
            payload = {"query": q, "k": int(k)}
            with st.spinner("Querying..."):
                try:
                    headers = {"Content-Type": "application/json"}
                    resp = requests.post(f"{API_URL}/query", data=json.dumps(payload), headers=headers)
                    resp.raise_for_status()
                    result = resp.json()

                    st.subheader("Answer")
                    st.write(result.get("answer") or "(no answer returned)")

                    st.subheader("Retrieved Documents")
                    docs = result.get("documents") or []
                    if not docs:
                        st.info("No documents were returned.")
                    for i, d in enumerate(docs):
                        meta = d.get("metadata", {})
                        with st.expander(f"Document #{i+1} - source: {meta.get('source', 'unknown')} id: {meta.get('id','N/A')}"):
                            st.markdown("**Metadata**")
                            st.json(meta)
                            st.markdown("**Content**")
                            st.write(d.get("content", ""))

                except requests.RequestException as e:
                    st.error(f"Query failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("API URL:")
st.sidebar.write(API_URL)
st.sidebar.markdown("Use `uvicorn backend.main:app --reload` to run the FastAPI server (from project root).")
