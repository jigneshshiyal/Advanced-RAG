import chromadb

client = chromadb.PersistentClient(path="./store_emb")

collection = client.get_or_create_collection(
    name="rag_collection", 
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200,
            "max_neighbors":32
        }
    }
)