import pandas as pd
from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")

def trials_to_docs(file_name="alzheimers_trials.csv"):
    df = pd.read_csv(PROCESSED_DATA_DIR / file_name)
    docs = []
    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        docs.append(text)
    return docs

if __name__ == "__main__":
    docs = trials_to_docs()
    print("Sample document:", docs[0][:500])