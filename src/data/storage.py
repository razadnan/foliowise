"""
src/data/storage.py
===================
Unified Storage Layer for FolioWise.

Supports CSV, Apache Parquet, and SQLite persistence for high-throughput
analytical queries and historical audits.
"""

import os
import sqlite3
import pandas as pd


class DataStorage:
    """
    Storage manager facilitating multi-format persistence across CSV,
    Parquet, and embedded SQLite.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, "data", "foliowise.db")
        os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Save DataFrame to Apache Parquet."""
        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            df.to_parquet(full_path, index=False, engine="fastparquet")
        except Exception:
            try:
                df.to_parquet(full_path, index=False)
            except Exception as e:
                print(f"Parquet export skipped: {e}")

    def load_parquet(self, path: str) -> pd.DataFrame:
        """Load DataFrame from Apache Parquet."""
        full_path = os.path.join(self.base_dir, path) if not os.path.isabs(path) else path
        try:
            return pd.read_parquet(full_path, engine="fastparquet")
        except Exception:
            return pd.read_parquet(full_path)

    def save_sqlite(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        """Save DataFrame into local SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def load_sqlite(self, query: str) -> pd.DataFrame:
        """Query local SQLite database into DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
