# src/preprocess.py
import pandas as pd
from pathlib import Path
from augment import augment_dataframe
import docx2txt
from PyPDF2 import PdfReader
import tiktoken  # optional: for token counting

# --------------------------
# Paths
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# --------------------------
# CSV Functions (Short-Context)
# --------------------------
def load_csv(file_name: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / file_name
    return pd.read_csv(path)

def filter_alzheimers_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows containing 'Alzheimer' in any column
    """
    mask = df.astype(str).apply(lambda x: x.str.contains("Alzheimer", case=False, na=False)).any(axis=1)
    return df[mask]

def save_processed(df: pd.DataFrame, file_name: str):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / file_name, index=False)

# --------------------------
# Full-Document Functions (Long-Context)
# --------------------------
def load_docx(file_path: Path) -> str:
    return docx2txt.process(str(file_path))

def load_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def preprocess_text(text: str) -> str:
    """
    Clean full document text:
    - Remove extra whitespaces
    - Normalize line breaks
    """
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split())
    return text

def count_tokens(text: str, model_name: str) -> int:
    """
    Count tokens for a given model using tiktoken
    """
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

# --------------------------
# Example script execution
# --------------------------
if __name__ == "__main__":
    # ----- CSV short-context workflow -----
    df = load_csv("ctg-studies.csv")
    df_filtered = filter_alzheimers_trials(df)

    df_aug = augment_dataframe(
    df_filtered,
    target_size=30000
)

    save_processed(df_aug, "alzheimers_trials_aug.csv")
    print(f"Processed {len(df_filtered)} Alzheimer’s rows; saved {len(df_aug)} rows after augmentation.")

    # ----- Full-document long-context workflow example -----
    example_pdf = RAW_DATA_DIR / "full_trial_example.pdf"
    if example_pdf.exists():
        raw_text = load_pdf(example_pdf)
        clean_text = preprocess_text(raw_text)
        num_tokens = count_tokens(clean_text, "gpt-4.1")
        print(f"Loaded full trial PDF: {example_pdf.name}, tokens: {num_tokens}")