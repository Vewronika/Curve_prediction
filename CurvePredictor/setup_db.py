import sqlite3
import pandas as pd
import os
import requests
import io
from config import DB_PATH


def setup_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    wiki_datasets = {
        "S&P 500": {
            "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "ticker_col": "Symbol",
            "name_col": "Security"
        },
        "Nasdaq 100": {
            "url": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
            "ticker_col": "Ticker",
            "name_col": "Company"
        },
        "Dow Jones": {
            "url": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",  # <-- NEW URL!
            "ticker_col": "Symbol",
            "name_col": "Company"
        }
    }


    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        cursor.execute('DROP TABLE IF EXISTS company_dataset')
        cursor.execute('DROP TABLE IF EXISTS dataset')
        cursor.execute('DROP TABLE IF EXISTS company')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS company
                       (
                           company_id INTEGER PRIMARY KEY AUTOINCREMENT,
                           ticker TEXT UNIQUE NOT NULL,
                           company_name TEXT NOT NULL
                       )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS dataset
                       (
                           dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                           dataset_name TEXT UNIQUE NOT NULL
                       )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS company_dataset
                       (
                           dataset_id INTEGER,
                           company_id INTEGER,
                           PRIMARY KEY (dataset_id, company_id),
                           FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
                           FOREIGN KEY (company_id) REFERENCES company(company_id)
                           )
                       ''')




        for dataset, config in wiki_datasets.items():
            print(f"Fetching {dataset}...")

            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                response = requests.get(config["url"], headers=headers)

                html_data = io.StringIO(response.text)
                tables = pd.read_html(html_data)

                company_table = None
                actual_ticker_col = None
                actual_name_col = None

                for table in tables:
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.get_level_values(0)

                    cols = [str(c) for c in table.columns]

                    t_col = next((c for c in cols if config["ticker_col"] in c), None)
                    n_col = next((c for c in cols if config["name_col"] in c), None)

                    if t_col and n_col:
                        company_table = table
                        actual_ticker_col = t_col
                        actual_name_col = n_col
                        break

                if company_table is None:
                    print(f"  -> Could not find a valid table for {dataset}, skipping.")
                    continue

                company_table[actual_ticker_col] = company_table[actual_ticker_col].str.replace('.', '-', regex=False)

                companies = list(zip(company_table[actual_ticker_col], company_table[actual_name_col]))

                cursor.executemany('''
                                   INSERT
                                   OR IGNORE INTO company (ticker, company_name)
                                            VALUES (?, ?)
                                   ''', companies)

                cursor.execute('''
                               INSERT
                               OR IGNORE INTO dataset (dataset_name) 
                                            VALUES (?)
                               ''', (dataset,))

                cursor.execute("SELECT dataset_id FROM dataset WHERE dataset_name = ?", (dataset,))
                dataset_id = cursor.fetchone()[0]

                tickers = [comp[0] for comp in companies]
                placeholders = ','.join(['?'] * len(tickers))
                cursor.execute(f"SELECT company_id FROM company WHERE ticker IN ({placeholders})", tickers)
                company_ids = cursor.fetchall()

                join_records = [(dataset_id, cid[0]) for cid in company_ids]
                cursor.executemany('''
                                   INSERT
                                   OR IGNORE INTO company_dataset (dataset_id, company_id)
                                                VALUES (?, ?)
                                   ''', join_records)
                print(f"Successfully loaded {len(companies)} companies for {dataset}.")

            except Exception as e:
                print(f"  -> Error scraping {dataset}: {e}")



if __name__ == "__main__":
    setup_database()