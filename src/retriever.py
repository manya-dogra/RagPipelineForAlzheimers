import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

EMBEDDINGS_DIR = Path("data/embeddings")

def load_index():
    index = faiss.read_index(str(EMBEDDINGS_DIR / "faiss_index.bin"))
    with open(EMBEDDINGS_DIR / "docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

def query_faiss(query, top_k=5, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    index, docs = load_index()
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = [docs[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    query = "What are the procedures for Alzheimer’s treatment?"
    results = query_faiss(query)
    print("\n".join(results))