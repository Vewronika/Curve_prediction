import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from data_acess import db_utils


class DatabaseHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = str(Path(self.temp_dir.name) / "test.sqlite3")
        self.db_path_patch = patch("data_acess.db_utils.DB_PATH", self.db_path)
        self.db_path_patch.start()
        self.addCleanup(self.db_path_patch.stop)

    def connect(self):
        @contextmanager
        def managed_connection():
            connection = sqlite3.connect(self.db_path)
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

        return managed_connection()

    def test_dataset_names_are_sorted_and_missing_dataset_table_is_empty(self):
        self.assertEqual(db_utils.fetch_all_dataset_names(), [])
        with self.connect() as connection:
            connection.execute("CREATE TABLE dataset (dataset_name TEXT)")
            connection.executemany(
                "INSERT INTO dataset(dataset_name) VALUES (?)",
                [("NASDAQ 100",), ("S&P 500",)],
            )

        self.assertEqual(db_utils.fetch_all_dataset_names(), ["NASDAQ 100", "S&P 500"])

    def test_current_schema_returns_companies_by_market_cap(self):
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE company (
                    company_id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    company_name TEXT
                );
                CREATE TABLE dataset (
                    dataset_id INTEGER PRIMARY KEY,
                    dataset_name TEXT
                );
                CREATE TABLE company_dataset (
                    dataset_id INTEGER,
                    company_id INTEGER,
                    company_rank INTEGER,
                    rank_type TEXT,
                    market_cap_usd REAL,
                    market_cap_as_of TEXT
                );
            """)
            connection.execute("INSERT INTO dataset VALUES (1, 'Example')")
            connection.executemany(
                "INSERT INTO company VALUES (?, ?, ?)",
                [(1, "AAA", "A"), (2, "BBB", "B"), (3, "CCC", "C")],
            )
            connection.executemany(
                "INSERT INTO company_dataset VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, 1, 1, "source_order", None, None),
                    (1, 2, 3, "source_order", 200.0, "2026-01-01"),
                    (1, 3, 2, "source_order", 100.0, "2026-01-01"),
                ],
            )

        companies = db_utils.fetch_companies_by_dataset("Example")

        self.assertEqual(companies["ticker"].tolist(), ["BBB", "CCC", "AAA"])
        self.assertEqual(companies.iloc[0]["market_cap_usd"], 200.0)

    def test_legacy_schema_uses_source_order_fallback(self):
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE company (
                    company_id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    company_name TEXT
                );
                CREATE TABLE dataset (
                    dataset_id INTEGER PRIMARY KEY,
                    dataset_name TEXT
                );
                CREATE TABLE company_dataset (
                    dataset_id INTEGER,
                    company_id INTEGER,
                    company_rank INTEGER
                );
            """)
            connection.execute("INSERT INTO dataset VALUES (1, 'Legacy')")
            connection.executemany(
                "INSERT INTO company VALUES (?, ?, ?)",
                [(1, "BBB", "B"), (2, "AAA", "A")],
            )
            connection.executemany(
                "INSERT INTO company_dataset VALUES (?, ?, ?)",
                [(1, 1, 2), (1, 2, 1)],
            )

        companies = db_utils.fetch_companies_by_dataset("Legacy")

        self.assertEqual(companies["ticker"].tolist(), ["AAA", "BBB"])
        self.assertEqual(companies["rank_type"].tolist(), ["source_order", "source_order"])
        self.assertTrue(companies["market_cap_usd"].isna().all())


if __name__ == "__main__":
    unittest.main()
