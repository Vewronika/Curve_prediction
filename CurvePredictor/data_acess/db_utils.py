import sqlite3
import pandas as pd
from config import DB_PATH

def fetch_all_dataset_names():
    query = "SELECT dataset_name FROM dataset"

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(query, connection)

    return df['dataset_name'].tolist()

def fetch_companies_by_dataset(dataset_name):
    query = """
            SELECT c.ticker, c.company_name, cd.company_rank
            FROM company c
                     JOIN company_dataset cd ON c.company_id = cd.company_id
                     JOIN dataset d ON cd.dataset_id = d.dataset_id
            WHERE d.dataset_name = ?
            ORDER BY cd.company_rank ASC
            """

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(query, connection, params=(dataset_name,))

    return df