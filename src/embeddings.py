import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from documents import trials_to_docs
from pathlib import Path
import pickle

EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def build_faiss_index(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    faiss.write_index(index, str(EMBEDDINGS_DIR / "faiss_index.bin"))
    with open(EMBEDDINGS_DIR / "docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    print(f"FAISS index built with {len(docs)} docs.")
    return index, docs

if __name__ == "__main__":
    docs = trials_to_docs()
    build_faiss_index(docs)