# src/embeddings.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from documents import trials_to_docs
from pathlib import Path
import pickle

EMBEDDINGS_DIR = Path("RagPipelineForAlzheimers/data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_VARIANT_QUALITY = 0.8   # filter low-quality synthetic rows

# ---------------------------------------------------------
def build_faiss_index(
    docs,
    model_name: str = MODEL_NAME,
):
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(EMBEDDINGS_DIR / "faiss_index.bin"))

    with open(EMBEDDINGS_DIR / "docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"[SUCCESS] FAISS index built with {len(docs)} documents")
    return index, docs


# ---------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "RagPipelineForAlzheimers/data/processed/alzheimers_trials_aug_v2.csv"

    print("[INFO] Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------------------
    # Filter low-quality synthetic rows
    # -----------------------------------------------------
    if "variant_quality" in df.columns:
        before = len(df)
        df = df[df["variant_quality"] >= MIN_VARIANT_QUALITY]
        after = len(df)
        print(f"[INFO] Filtered variants: {before} → {after}")

    # -----------------------------------------------------
    # Convert trials to documents
    # -----------------------------------------------------
    docs = trials_to_docs(
        file_name=Path(DATA_PATH).name
    )

    # -----------------------------------------------------
    # Build FAISS index
    # -----------------------------------------------------
    build_faiss_index(docs)