import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

def load_index():
    index = faiss.read_index(str(EMBEDDINGS_DIR / "faiss_index.bin"))
    with open(EMBEDDINGS_DIR / "docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

def unique_by_source_nct(texts, max_items=5):
    """
    Collapse near-duplicate results by their source study identifier.
    Assumes each concatenated doc string contains either `source_nct: ...`
    or `NCT Number: ...`. Falls back to hashing the text if absent.
    """
    seen = set()
    unique = []
    for t in texts:
        src = None
        for part in t.split(" | "):
            part = part.strip()
            if part.startswith("source_nct:"):
                src = part.split("source_nct:")[1].strip()
                break
            if part.startswith("NCT Number:"):
                src = part.split("NCT Number:")[1].strip()
        if src is None:
            src = str(hash(t))
        if src in seen:
            continue
        seen.add(src)
        unique.append(t)
        if len(unique) >= max_items:
            break
    return unique

def query_faiss(
    query: str,
    top_k: int = 10,                  
    model_name: str = "all-MiniLM-L6-v2",
    dedup_by_source: bool = True,
    overretrieve_factor: int = 4   
):
    """
    Encode query, search FAISS, then optionally de-duplicate results by source_nct.
    Over-retrieves k*overretrieve_factor candidates to provide headroom for de-dup.
    """
    model = SentenceTransformer(model_name)
    index, docs = load_index()
    query_vec = model.encode([query], convert_to_numpy=True)

    raw_k = max(top_k * max(1, overretrieve_factor), top_k)
    distances, indices = index.search(query_vec, raw_k)

    flat_indices = indices[0].tolist() if hasattr(indices, "__getitem__") else []
    candidates = [docs[i] for i in flat_indices if 0 <= i < len(docs)]

    if dedup_by_source:
        results = unique_by_source_nct(candidates, max_items=top_k)
        if len(results) < top_k:
            seen_texts = set(results)
            for c in candidates:
                if c not in seen_texts:
                    results.append(c)
                    seen_texts.add(c)
                    if len(results) >= top_k:
                        break
    else:
        results = candidates[:top_k]

    return results

if __name__ == "__main__":
    q = "What are the procedures for Alzheimer’s treatment?"
    out = query_faiss(q, top_k=10, dedup_by_source=True, overretrieve_factor=4)
    print("\n--- Results ---\n")
    for i, r in enumerate(out, 1):
        print(f"[{i}] {r[:300]}...\n")
