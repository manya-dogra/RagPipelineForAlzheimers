# src/documents.py
import pandas as pd
from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")

ALLOWLIST = [
    "NCT Number", "Study Title", "Study URL", "Study Status",
    "Brief Summary", "Conditions", "Interventions",
    "Primary Outcome Measures", "Secondary Outcome Measures", "Other Outcome Measures",
    "Phases", "Enrollment", "Funder Type", "Study Type", "Study Design",
    "Start Date", "Primary Completion Date", "Completion Date",
    "source_nct", "variant_id", "variant_method", "variant_quality"
]

def trials_to_docs(file_name="alzheimers_trials_aug.csv"):
    df = pd.read_csv(PROCESSED_DATA_DIR / file_name)
    # keep only columns that exist
    use_cols = [c for c in ALLOWLIST if c in df.columns]
    docs = []
    for _, row in df.iterrows():
        parts = []
        for col in use_cols:
            val = row[col]
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        docs.append(" | ".join(parts))
    return docs
