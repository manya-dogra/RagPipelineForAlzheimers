import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

def load_csv(file_name: str) -> pd.DataFrame:
    """Load a CSV from data/raw/"""
    path = RAW_DATA_DIR / file_name
    return pd.read_csv(path)

def filter_alzheimers_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Alzheimer’s-related rows"""
    mask = df.astype(str).apply(lambda x: x.str.contains("Alzheimer", case=False, na=False)).any(axis=1)
    return df[mask]

def save_processed(df: pd.DataFrame, file_name: str):
    """Save processed CSV to data/processed/"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / file_name, index=False)

if __name__ == "__main__":
    df = load_csv("ctg-studies.csv")   # replace with your filename
    df_filtered = filter_alzheimers_trials(df)
    save_processed(df_filtered, "alzheimers_trials.csv")
    print(f"Processed {len(df_filtered)} Alzheimer’s rows saved.")