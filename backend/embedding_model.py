from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time
from tqdm import tqdm
from config import GEMINI_API_KEY

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def gemini_embedding_batch(texts, batch_size=2):
    result_emb = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        result_emb.extend(embeddings.embed_documents(batch))
        time.sleep(2)
    return result_emb