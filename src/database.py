"""Load the cleaned data into a SQLite database (data/processed/heart.db).

Run from the project root after `python -m src.cleaning`:

    python -m src.database
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.cleaning import CATEGORY_LABELS, CLEAN_PATH, PROCESSED_DIR
from src.data_loading import PROJECT_ROOT

DB_PATH = PROCESSED_DIR / "heart.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

# Lookup table name -> column in `patients` whose codes it describes
LOOKUP_TABLES = {
    "chest_pain_types": "cp",
    "resting_ecg_results": "restecg",
    "st_slopes": "slope",
    "thal_results": "thal",
}

PATIENT_COLUMNS = [
    "patient_id", "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the database with foreign-key enforcement switched on."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build_database(clean_df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """(Re)create heart.db from the schema and load the cleaned rows.

    `target` is not inserted: the schema computes it from `num`.
    Any row that breaks a CHECK or FOREIGN KEY constraint raises an error.
    """
    schema = SCHEMA_PATH.read_text()
    with connect(db_path) as conn:
        conn.executescript(schema)
        for table, column in LOOKUP_TABLES.items():
            conn.executemany(
                f"INSERT INTO {table} (code, label) VALUES (?, ?)",
                CATEGORY_LABELS[column].items(),
            )
        placeholders = ", ".join("?" for _ in PATIENT_COLUMNS)
        rows = clean_df[PATIENT_COLUMNS].astype(object).itertuples(index=False)
        conn.executemany(
            f"INSERT INTO patients ({', '.join(PATIENT_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        check_against(conn, clean_df)
    conn.close()


def check_against(conn: sqlite3.Connection, clean_df: pd.DataFrame) -> None:
    """Confirm the database holds exactly what the cleaned CSV holds."""
    n_rows = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    n_labeled = conn.execute("SELECT COUNT(*) FROM patients_labeled").fetchone()[0]
    n_disease = conn.execute("SELECT SUM(target) FROM patients").fetchone()[0]
    if n_rows != len(clean_df) or n_labeled != len(clean_df):
        raise RuntimeError(f"Row count mismatch: db={n_rows}, view={n_labeled}, csv={len(clean_df)}")
    if n_disease != clean_df["target"].sum():
        raise RuntimeError("target column in the database does not match the CSV")


def main() -> None:
    clean_df = pd.read_csv(CLEAN_PATH)
    build_database(clean_df)
    with connect() as conn:
        summary = pd.read_sql(
            "SELECT target, COUNT(*) AS patients FROM patients GROUP BY target", conn
        )
    conn.close()
    print(f"Wrote {DB_PATH.relative_to(PROJECT_ROOT)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
