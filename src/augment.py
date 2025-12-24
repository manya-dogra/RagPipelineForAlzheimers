# src/augment.py
import pandas as pd
import random

AUG_FIELDS = [
    "Brief Summary",
    "Primary Outcome Measures",
    "Secondary Outcome Measures",
    "Other Outcome Measures",
    "Conditions",
    "Interventions",
]

def simple_paraphrase(text: str, level: int = 1) -> str:
    """
    Lightweight paraphrasing that preserves clinical meaning.
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

    if level == 1:
        variants = [
            out,
            f"In this study, {out}",
        ]
    elif level == 2:
        variants = [
            f"The objective of this trial is as follows: {out}",
            f"{out} This approach follows standard clinical protocols.",
        ]
    else:
        variants = [
            f"{out} No deviation from the original methodology is intended."
        ]

    return random.choice(variants)


def augment_dataframe(
    df: pd.DataFrame,
    target_size: int,
    max_variants_per_record: int = 5,
) -> pd.DataFrame:
    """
    Oversample dataframe until target_size is reached.
    Adds lineage and quality metadata.
    """

    df = df.copy()

    # ---- lineage columns ----
    if "source_nct" not in df.columns:
        df["source_nct"] = df.get("NCT Number")

    if "variant_id" not in df.columns:
        df["variant_id"] = 0

    if "variant_method" not in df.columns:
        df["variant_method"] = "original"

    if "variant_quality" not in df.columns:
        df["variant_quality"] = 1.0

    original_size = len(df)

    if target_size <= original_size:
        print(f"[INFO] Dataset already has {original_size} rows. No oversampling needed.")
        return df

    augmented_rows = []
    original_df = df.copy()

    while len(df) + len(augmented_rows) < target_size:
        row = original_df.sample(1).iloc[0]
        level = random.choice([1, 2, 3])

        new_row = row.copy()
        for field in AUG_FIELDS:
            if field in new_row and pd.notna(new_row[field]):
                new_row[field] = simple_paraphrase(
                    new_row[field],
                    level=level
                )

        new_row["source_nct"] = row["source_nct"]
        new_row["variant_id"] = random.randint(1, max_variants_per_record)
        new_row["variant_method"] = f"paraphrase_level_{level}"
        new_row["variant_quality"] = round(1.0 - (0.1 * level), 2)

        augmented_rows.append(new_row)

    df_aug = pd.DataFrame(augmented_rows, columns=df.columns)
    df_out = pd.concat([df, df_aug], ignore_index=True)

    print(
        f"[SUCCESS] Oversampling complete: "
        f"{original_size} → {len(df_out)} rows"
    )

    return df_out


# ------------------------------------------------------------------
# RUN MODE (allows: python src/augment.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_PATH = "RagPipelineForAlzheimers/data/processed/alzheimers_trials_aug.csv"
    OUTPUT_PATH = "RagPipelineForAlzheimers/data/processed/alzheimers_trials_aug_v2.csv"
    TARGET_SIZE = 30000

    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Original dataset size: {len(df)}")

    df_oversampled = augment_dataframe(
        df,
        target_size=TARGET_SIZE
    )

    df_oversampled.to_csv(OUTPUT_PATH, index=False)

    print(f"[DONE] Oversampled dataset saved to: {OUTPUT_PATH}")