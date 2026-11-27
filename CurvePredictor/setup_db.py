"""Refresh current index membership without dropping existing database tables.

Sources are public index constituent pages. The stored rank is source order, not
market-cap rank. Market-cap fields are reserved for a separately refreshed,
explicitly dated enrichment step; they are not fabricated here.
"""
from __future__ import annotations

import io
import os
import sqlite3
from datetime import date

import pandas as pd
import requests

from config import DB_PATH

DATASETS = {
    "S&P 500": {"url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "ticker_col": "Symbol", "name_col": "Security"},
    "S&P 400": {"url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "ticker_col": "Symbol", "name_col": "Security"},
    "S&P 600": {"url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "ticker_col": "Symbol", "name_col": "Security"},
    "Nasdaq 100": {"url": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "ticker_col": "Ticker", "name_col": "Company"},
    "Dow Jones": {"url": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies", "ticker_col": "Symbol", "name_col": "Company"},
}
USER_AGENT = "CurvePredictor academic project contact: repository maintainer"


def _flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    if isinstance(table.columns, pd.MultiIndex):
        table = table.copy()
        table.columns = [" ".join(str(part) for part in col if str(part) != "nan").strip() for col in table.columns]
    return table


def _normalise_column_name(value: object) -> str:
    """Normalize HTML table headers so punctuation/case differences do not matter."""
    return " ".join(str(value).replace("\xa0", " ").strip().lower().split())


def _find_table(tables, ticker_hint: str, name_hint: str) -> pd.DataFrame:
    # Wikipedia's labels differ by index and sometimes change. Prefer known aliases,
    # while still requiring both a ticker-like and company-name column in one table.
    ticker_aliases = {
        "symbol", "ticker", "ticker symbol", "ticker-symbol", "nasdaq symbol",
        "stock symbol", "symbol (ticker)",
    }
    name_aliases = {
        "company", "company name", "security", "name", "constituent",
        "company/security", "constituent name",
    }
    ticker_aliases.add(_normalise_column_name(ticker_hint))
    name_aliases.add(_normalise_column_name(name_hint))

    for original in tables:
        table = _flatten_columns(original)
        cols = list(table.columns)
        normalized = {_normalise_column_name(col): col for col in cols}

        ticker_col = next((normalized[a] for a in ticker_aliases if a in normalized), None)
        name_col = next((normalized[a] for a in name_aliases if a in normalized), None)
        if ticker_col is None or name_col is None or ticker_col == name_col:
            continue

        result = table[[ticker_col, name_col]].copy()
        result.columns = ["ticker", "company_name"]
        result["ticker"] = (
            result["ticker"].astype(str).str.strip()
            .str.replace(".", "-", regex=False)
        )
        result["company_name"] = result["company_name"].astype(str).str.strip()
        invalid = {"", "nan", "none", "nat"}
        result = result[
            ~result["ticker"].str.lower().isin(invalid)
            & ~result["company_name"].str.lower().isin(invalid)
        ]
        # Avoid accidentally accepting an unrelated table with two matching headers.
        result = result[result["ticker"].str.match(r"^[A-Za-z0-9.^$-]+$", na=False)]
        result = result.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
        if len(result) >= 20:
            return result

    raise ValueError(
        f"Could not identify a constituent table with ticker/name columns. "
        f"Expected ticker aliases {sorted(ticker_aliases)} and name aliases {sorted(name_aliases)}."
    )


def _create_or_migrate_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS company (
                                               company_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                               ticker TEXT UNIQUE NOT NULL,
                                               company_name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dataset (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT UNIQUE NOT NULL,
            source_url TEXT,
            source_as_of TEXT,
            membership_type TEXT NOT NULL DEFAULT 'current'
        );
        CREATE TABLE IF NOT EXISTS company_dataset (
            dataset_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            company_rank INTEGER NOT NULL,
            rank_type TEXT NOT NULL DEFAULT 'source_order',
            market_cap_usd REAL,
            market_cap_as_of TEXT,
            PRIMARY KEY (dataset_id, company_id),
            FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
            FOREIGN KEY (company_id) REFERENCES company(company_id)
        );
    """)
    # Migrate existing installations in place. Never DROP user data/tables.
    dataset_cols = {r[1] for r in connection.execute("PRAGMA table_info(dataset)")}
    for name, sql in [("source_url", "ALTER TABLE dataset ADD COLUMN source_url TEXT"),
                      ("source_as_of", "ALTER TABLE dataset ADD COLUMN source_as_of TEXT"),
                      ("membership_type", "ALTER TABLE dataset ADD COLUMN membership_type TEXT NOT NULL DEFAULT 'current'")]:
        if name not in dataset_cols:
            connection.execute(sql)
    membership_cols = {r[1] for r in connection.execute("PRAGMA table_info(company_dataset)")}
    for name, sql in [("rank_type", "ALTER TABLE company_dataset ADD COLUMN rank_type TEXT NOT NULL DEFAULT 'source_order'"),
                      ("market_cap_usd", "ALTER TABLE company_dataset ADD COLUMN market_cap_usd REAL"),
                      ("market_cap_as_of", "ALTER TABLE company_dataset ADD COLUMN market_cap_as_of TEXT")]:
        if name not in membership_cols:
            connection.execute(sql)


def setup_database() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    with sqlite3.connect(DB_PATH) as connection:
        _create_or_migrate_schema(connection)
        for dataset_name, spec in DATASETS.items():
            print(f"Fetching {dataset_name}...")
            try:
                response = requests.get(spec["url"], headers=headers, timeout=25)
                response.raise_for_status()
                tables = pd.read_html(io.StringIO(response.text))
                constituents = _find_table(tables, spec["ticker_col"], spec["name_col"])
                dataset_as_of = date.today().isoformat()
                connection.execute(
                    "INSERT INTO dataset(dataset_name, source_url, source_as_of, membership_type) VALUES (?, ?, ?, 'current') "
                    "ON CONFLICT(dataset_name) DO UPDATE SET source_url=excluded.source_url, source_as_of=excluded.source_as_of, membership_type='current'",
                    (dataset_name, spec["url"], dataset_as_of),
                )
                dataset_id = connection.execute("SELECT dataset_id FROM dataset WHERE dataset_name=?", (dataset_name,)).fetchone()[0]
                connection.executemany(
                    "INSERT INTO company(ticker, company_name) VALUES (?, ?) "
                    "ON CONFLICT(ticker) DO UPDATE SET company_name=excluded.company_name",
                    list(constituents.itertuples(index=False, name=None)),
                )
                ticker_ids = {ticker: company_id for ticker, company_id in connection.execute("SELECT ticker, company_id FROM company")}
                # Replace only membership rows for this dataset, not the company catalog or other datasets.
                connection.execute("DELETE FROM company_dataset WHERE dataset_id=?", (dataset_id,))
                rows = [(dataset_id, ticker_ids[row.ticker], rank, "source_order", None, None)
                        for rank, row in enumerate(constituents.itertuples(index=False), start=1)]
                connection.executemany(
                    "INSERT INTO company_dataset(dataset_id, company_id, company_rank, rank_type, market_cap_usd, market_cap_as_of) VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                connection.commit()
                print(f"Loaded {len(rows)} current constituents for {dataset_name}; ranking is source order, not market cap.")
            except Exception as exc:
                connection.rollback()
                print(f"ERROR: could not refresh {dataset_name}: {exc}")
        print("Database refresh finished. Existing tables were preserved; failed sources were left unchanged.")


if __name__ == "__main__":
    setup_database()
