"""Load the UCI Cleveland Heart Disease dataset with proper column names."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"

# The raw file has no header row; names follow the UCI documentation.
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. Missing values in the file are written as '?'."""
    return pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary `target`: 1 if any heart disease (num 1-4), else 0."""
    out = df.copy()
    out["target"] = (out["num"] > 0).astype(int)
    return out
