"""Read-only helpers for the local universe metadata database."""
from contextlib import closing
import sqlite3
import pandas as pd
from config import DB_PATH


def fetch_all_dataset_names():
    with closing(sqlite3.connect(DB_PATH)) as connection:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dataset'", connection
        )
        if tables.empty:
            return []
        return pd.read_sql_query(
            "SELECT dataset_name FROM dataset ORDER BY dataset_name", connection
        )["dataset_name"].tolist()


def fetch_companies_by_dataset(dataset_name):
    query = """
        SELECT c.ticker, c.company_name, cd.company_rank,
               cd.rank_type, cd.market_cap_usd, cd.market_cap_as_of
        FROM company c
        JOIN company_dataset cd ON c.company_id = cd.company_id
        JOIN dataset d ON cd.dataset_id = d.dataset_id
        WHERE d.dataset_name = ?
        ORDER BY
            CASE WHEN cd.market_cap_usd IS NULL THEN 1 ELSE 0 END,
            cd.market_cap_usd DESC,
            cd.company_rank ASC,
            c.ticker ASC
    """
    with closing(sqlite3.connect(DB_PATH)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(company_dataset)")}
        # Read older databases safely until setup_db.py is run to migrate them.
        if not {"rank_type", "market_cap_usd", "market_cap_as_of"}.issubset(columns):
            query = """
                SELECT c.ticker, c.company_name, cd.company_rank,
                       'source_order' AS rank_type,
                       NULL AS market_cap_usd, NULL AS market_cap_as_of
                FROM company c
                JOIN company_dataset cd ON c.company_id = cd.company_id
                JOIN dataset d ON cd.dataset_id = d.dataset_id
                WHERE d.dataset_name = ?
                ORDER BY cd.company_rank ASC, c.ticker ASC
            """
        return pd.read_sql_query(query, connection, params=(dataset_name,))
