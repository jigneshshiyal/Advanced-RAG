from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from vectordb import client
from embedding_model import embeddings
from llm import model as llm
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from collections import defaultdict
import numpy as np
from scipy.special import softmax

# --- Setup ---
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # precise reranker
db = Chroma(collection_name="rag_collection", client=client, embedding_function=embeddings)

retriever_sim = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5})

# --- Prompt Template ---
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the retrieved documents to answer the user query.
Cite relevant document metadata (like title, id, or source) when appropriate.

Context:
{context}

Question: {question}
""")

# --- Utility ---
def normalize(arr):
    arr = np.array(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)

# --- Hybrid retrieval (similarity + MMR + BM25) ---
def hybrid_retrieve(query, chroma_db, embeddings, k=10, alpha=0.6, weight_mmr=0.3):
    """
    Combines similarity, MMR, and BM25 retrieval into a normalized hybrid score.
    alpha: weight for semantic similarity
    weight_mmr: weight for MMR diversity
    (1 - alpha - weight_mmr): weight for BM25 lexical relevance
    """

    # Step 1: similarity docs
    sim_results = chroma_db._collection.query(
        query_embeddings=embeddings.embed_query(query),
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )
    sim_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(sim_results["documents"][0], sim_results["metadatas"][0])
    ]
    sim_dists = np.array(sim_results["distances"][0])
    sim_scores = 1 / (1 + sim_dists)  # convert distance → similarity
    sim_norm = normalize(sim_scores)

    # Step 2: mmr docs
    mmr_docs = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}
    ).invoke(query)

    # Step 3: BM25
    corpus = [doc.page_content for doc in sim_docs]
    tokenized_corpus = [c.split() for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = normalize(bm25.get_scores(query.split()))

    # Step 4: combine scores using stable doc IDs
    doc_scores = defaultdict(float)
    doc_id_map = {id(doc): doc for doc in sim_docs}

    # semantic similarity
    for i, doc in enumerate(sim_docs):
        doc_scores[id(doc)] += alpha * sim_norm[i]

    # MMR reciprocal-rank
    for i, doc in enumerate(mmr_docs):
        key = id(doc)
        doc_scores[key] += weight_mmr / (i + 1)

    # BM25 lexical
    for i, score in enumerate(bm25_scores):
        doc_scores[id(sim_docs[i])] += (1 - alpha - weight_mmr) * score

    # Step 5: rank and attach metadata
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ranked_docs = []
    for key, score in ranked:
        doc = doc_id_map.get(key)
        if doc:
            doc.metadata["hybrid_score"] = float(score)
            ranked_docs.append(doc)

    return ranked_docs

# --- Cross-encoder reranking ---
def rerank_with_cross_encoder(query, candidates, weight_cross=0.7, weight_hybrid=0.3):
    """
    Re-rank top documents using cross-encoder.
    Combines normalized cross-encoder scores and hybrid retrieval scores.
    """
    if not candidates:
        return []

    pairs = [[query, c.page_content] for c in candidates]
    cross_scores = cross_model.predict(pairs)
    cross_norm = softmax(cross_scores)

    hybrid_arr = np.array([c.metadata.get("hybrid_score", 0.0) for c in candidates])
    hybrid_norm = normalize(hybrid_arr)

    combined = weight_cross * cross_norm + weight_hybrid * hybrid_norm

    for c, cs, cn, hn, final in zip(
        candidates, cross_scores, cross_norm, hybrid_norm, combined
    ):
        c.metadata["cross_score"] = float(cs)
        c.metadata["cross_norm"] = float(cn)
        c.metadata["hybrid_norm"] = float(hn)
        c.metadata["final_score"] = float(final)

    ranked = sorted(candidates, key=lambda x: x.metadata["final_score"], reverse=True)
    return ranked

# --- Main answer function ---
def answer_query(query):
    """
    Retrieves, ranks, and answers a user query with full traceability.
    Returns both the generated answer and ranked documents with metadata.
    """
    # Step 1: hybrid retrieval
    candidate_docs = hybrid_retrieve(query, db, embeddings, k=20)

    # Step 2: rerank with cross-encoder
    reranked = rerank_with_cross_encoder(query, candidate_docs)[:5]

    # Step 3: build context
    combined_context = "\n\n---\n\n".join(
        [
            f"Source: {d.metadata.get('source', 'unknown')} | "
            f"ID: {d.metadata.get('id', 'N/A')} | "
            f"Hybrid: {d.metadata.get('hybrid_score', 0):.3f} | "
            f"Cross: {d.metadata.get('cross_score', 0):.3f}\n"
            f"{d.page_content}"
            for d in reranked
        ]
    )

    # Step 4: generate final answer
    response = llm.invoke(prompt.format(context=combined_context, question=query))

    # Step 5: return answer + traceable ranked docs
    return {
        "answer": response,
        "documents": [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in reranked
        ]
    }
