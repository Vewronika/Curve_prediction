import sqlite3
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime
from config import DB_PATH

def validate_dataframe(df):
    required_columns = {"curve_name", "time_step", "value"}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain these columns: {required_columns}")

    return df

def parse_uploaded_csv(csv_file):
    df = pd.read_csv(csv_file)
    return validate_dataframe(df)


def get_sliced_portfolio(all_tickers, selected_tickers, end_date, interval="1d"):
    """
    Fetches the giant dataset using the cached function, then cleanly slices it.
    This keeps data manipulation logic out of the Streamlit UI.
    """
    # This calls your cached function, so it's lightning fast!
    giant_df = fetch_stock_data(all_tickers, end_date, interval)

    # Safely slice it
    valid_tickers = [t for t in selected_tickers if t in giant_df.columns]
    return giant_df[valid_tickers]


@st.cache_data(ttl="15m", max_entries=10)
def fetch_stock_data(tickers: list, end_date, interval: str = "1d") -> pd.DataFrame:
    tickers_string = " ".join(tickers)

    if isinstance(end_date, str):
        end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_date_obj = end_date

    if interval == "1h":
        start_date_obj = end_date_obj - datetime.timedelta(days=30)
    elif interval == "1wk":
        start_date_obj = end_date_obj - datetime.timedelta(days=1500)
    else:
        start_date_obj = end_date_obj - datetime.timedelta(days=365)

    start_str = start_date_obj.strftime("%Y-%m-%d")
    end_str = end_date_obj.strftime("%Y-%m-%d")

    df = yf.download(
        tickers_string,
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=True
    )

    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        elif 'Adj Close' in df.columns.get_level_values(0):
            df = df['Adj Close']
    else:
        if 'Close' in df.columns:
            df = df[['Close']]
            df.columns = tickers

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    df = df.dropna(how='all')
    df = df.ffill()
    df = df.dropna(axis=1)

    return df