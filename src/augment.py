# src/augment.py
import pandas as pd
from typing import List, Dict
import random

AUG_FIELDS = [
    "Brief Summary",
    "Primary Outcome Measures",
    "Secondary Outcome Measures",
    "Other Outcome Measures",
    "Conditions",
    "Interventions",
]

def simple_paraphrase(text: str) -> str:
    """
    Very light, deterministic paraphrase stub.
    Replace with your preferred model or back-translation pipeline.
    Must preserve entities and avoid adding new claims.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    replacements = {
        "study": "trial",
        "participants": "subjects",
        "evaluate": "assess",
        "assess": "evaluate",
        "treatment": "therapy",
        "Alzheimer": "Alzheimer’s",
    }
    out = text
    for a, b in replacements.items():
        out = out.replace(a, b)
    variants = [
        out,
        f"In this context, {out}",
        f"{out} The objective remains unchanged.",
    ]
    return random.choice(variants)

def paraphrase_row(row: pd.Series, method: str = "simple_paraphrase") -> pd.Series:
    row_copy = row.copy()
    for f in AUG_FIELDS:
        if f in row_copy and pd.notna(row_copy[f]):
            row_copy[f] = simple_paraphrase(row_copy[f])
    row_copy["variant_method"] = method
    return row_copy

def augment_dataframe(
    df: pd.DataFrame,
    variants_per_record: int = 1,
    methods: List[str] = None,
) -> pd.DataFrame:
    """
    Add 1..N paraphrased variants per original row as new rows.
    Adds lineage columns: source_nct, variant_id, variant_method, variant_quality.
    """
    methods = methods or ["simple_paraphrase"]
    df = df.copy()

    if "source_nct" not in df.columns:
        df["source_nct"] = df.get("NCT Number", pd.Series([None]*len(df)))
    if "variant_id" not in df.columns:
        df["variant_id"] = 0
    if "variant_method" not in df.columns:
        df["variant_method"] = "original"
    if "variant_quality" not in df.columns:
        df["variant_quality"] = 1.0  # default pass

    augmented_rows = []
    for _, row in df.iterrows():
        for v in range(1, variants_per_record + 1):
            method = random.choice(methods)
            new_row = paraphrase_row(row, method=method)
            new_row["source_nct"] = row.get("NCT Number", row.get("source_nct", None))
            new_row["variant_id"] = v
            new_row["variant_quality"] = 1.0
            augmented_rows.append(new_row)

    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows, columns=df.columns)
        df_out = pd.concat([df, df_aug], ignore_index=True)
    else:
        df_out = df

    return df_out
