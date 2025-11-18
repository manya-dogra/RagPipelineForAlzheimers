# src/preprocess.py
import pandas as pd
from pathlib import Path
from augment import augment_dataframe

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

def load_csv(file_name: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / file_name
    return pd.read_csv(path)

def filter_alzheimers_trials(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.astype(str).apply(lambda x: x.str.contains("Alzheimer", case=False, na=False)).any(axis=1)
    return df[mask]

def save_processed(df: pd.DataFrame, file_name: str):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / file_name, index=False)

if __name__ == "__main__":
    df = load_csv("ctg-studies.csv")
    df_filtered = filter_alzheimers_trials(df)

    df_aug = augment_dataframe(
        df_filtered,
        variants_per_record=2,
        methods=["simple_paraphrase"]  
    )

    save_processed(df_aug, "alzheimers_trials_aug.csv")
    print(f"Processed {len(df_filtered)} Alzheimer’s rows; saved {len(df_aug)} rows after augmentation.")
