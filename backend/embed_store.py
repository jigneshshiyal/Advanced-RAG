from embedding_model import gemini_embedding_batch
from vectordb import collection

def embed_and_store(docs):
    # chunking 
    texts = [doc['text'] for doc in docs]
    metadatas = [doc['metadata'] for doc in docs]
    embs = gemini_embedding_batch(texts)

    
    ids = [doc['id'] for doc in docs]
    
    collection.add(
        documents=texts,
        metadatas=metadatas,
        embeddings=embs,
        ids=ids
    )