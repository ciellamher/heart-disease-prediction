"""Clean the raw Cleveland data and produce model-ready encodings.

Run from the project root to write the cleaned files to data/processed/:

    python -m src.cleaning
"""

from pathlib import Path

import pandas as pd

from src.data_loading import PROJECT_ROOT, add_target, load_raw

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "heart_clean.csv"
ENCODED_PATH = PROCESSED_DIR / "heart_encoded.csv"

# Code -> label for every categorical column (from data/DATA_DICTIONARY.md).
# Single source of truth: used for validation, one-hot column names and the
# SQLite lookup tables.
CATEGORY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {
        1: "typical_angina",
        2: "atypical_angina",
        3: "non_anginal_pain",
        4: "asymptomatic",
    },
    "fbs": {0: "false", 1: "true"},
    "restecg": {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed_defect", 7: "reversible_defect"},
}

# Integer-valued columns. Everything is read as float because of the `?`
# markers, so these are cast back once missing rows are gone.
INT_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "slope", "ca", "thal", "num",
]

# Multi-class nominal columns that get one-hot encoded for modelling.
# Binary columns (sex, fbs, exang) are already 0/1; ca is a count (0-3).
ONE_HOT_COLUMNS = ["cp", "restecg", "slope", "thal"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    Steps:
      1. Add `patient_id` = the row's line number in the raw file (1-based),
         so every cleaned row can be traced back to the source.
      2. Drop the 6 rows with a missing `ca` or `thal` (about 2% of the data).
         Imputing a category code for a diagnostic test result would invent
         clinical findings, and losing 6 rows barely changes the class balance.
      3. Cast integer-valued columns from float to int.
      4. Add the binary `target` (1 = any disease).
      5. Check every value is in its documented range.
    """
    out = df.copy()
    out.insert(0, "patient_id", range(1, len(out) + 1))
    out = out.dropna(subset=["ca", "thal"]).reset_index(drop=True)
    out[INT_COLUMNS] = out[INT_COLUMNS].astype(int)
    out = add_target(out)
    validate(out)
    return out


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the cleaned data breaks any documented rule."""
    problems = []
    if df.isna().any().any():
        problems.append("missing values remain")
    if df["patient_id"].duplicated().any():
        problems.append("duplicate patient_id")
    for col, labels in CATEGORY_LABELS.items():
        bad = ~df[col].isin(labels)
        if bad.any():
            problems.append(f"{col}: unexpected codes {sorted(df.loc[bad, col].unique())}")
    if not df["ca"].between(0, 3).all():
        problems.append("ca outside 0-3")
    if not df["num"].between(0, 4).all():
        problems.append("num outside 0-4")
    if (df["oldpeak"] < 0).any():
        problems.append("negative oldpeak")
    if problems:
        raise ValueError("Cleaned data failed validation: " + "; ".join(problems))


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the nominal categories for modelling.

    `cp=4` becomes `cp_asymptomatic=1`, and so on. All levels are kept (no
    dropped baseline) so each column reads directly as "has this finding".
    `num` is removed because `target` is derived from it (it would leak the
    answer); `patient_id` is kept as an identifier, not a feature.
    """
    out = df.drop(columns=["num"]).copy()
    for col in ONE_HOT_COLUMNS:
        labelled = pd.Categorical(
            out[col].map(CATEGORY_LABELS[col]),
            categories=list(CATEGORY_LABELS[col].values()),
        )
        dummies = pd.get_dummies(labelled, prefix=col, dtype=int)
        dummies.index = out.index
        position = out.columns.get_loc(col)
        out = out.drop(columns=[col])
        for offset, name in enumerate(dummies.columns):
            out.insert(position + offset, name, dummies[name])
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cleaned = clean(raw)
    encoded = encode(cleaned)
    cleaned.to_csv(CLEAN_PATH, index=False)
    encoded.to_csv(ENCODED_PATH, index=False)
    print(f"Raw rows: {len(raw)}  ->  cleaned rows: {len(cleaned)}")
    print(f"Wrote {CLEAN_PATH.relative_to(PROJECT_ROOT)} {cleaned.shape}")
    print(f"Wrote {ENCODED_PATH.relative_to(PROJECT_ROOT)} {encoded.shape}")


if __name__ == "__main__":
    main()
