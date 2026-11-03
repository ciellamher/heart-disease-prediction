"""Load the named queries in sql/analysis.sql and run them against heart.db."""

import re
from pathlib import Path

import pandas as pd

from src.data_loading import PROJECT_ROOT
from src.database import connect

ANALYSIS_PATH = PROJECT_ROOT / "sql" / "analysis.sql"

_NAME_LINE = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def load_queries(path: Path = ANALYSIS_PATH) -> dict[str, str]:
    """Split the file on its `-- name: <query_name>` lines.

    Returns {query_name: sql_text} in file order.
    """
    text = path.read_text()
    parts = _NAME_LINE.split(text)
    # parts = [header, name1, body1, name2, body2, ...]
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def run_query(name: str, path: Path = ANALYSIS_PATH) -> pd.DataFrame:
    """Run one named query and return the result as a DataFrame."""
    sql = load_queries(path)[name]
    conn = connect()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()
