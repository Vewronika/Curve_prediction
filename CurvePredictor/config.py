import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "data.sqlite3")
TABLE_DATASET = "dataset"
TABLE_CURVE_POINT = "curve_point"
