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



@st.cache_data(ttl="15m", max_entries=10)
def fetch_stock_data(tickers: list, end_date, interval: str = "1d") -> pd.DataFrame:
    tickers_string = " ".join(tickers)

    # 1. Ensure end_date is a proper datetime date object
    if isinstance(end_date, str):
        end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_date_obj = end_date

    # 2. Calculate the exact START date based on the interval
    if interval == "1h":
        start_date_obj = end_date_obj - datetime.timedelta(days=30)  # 30 days is safe for 1h limit
    elif interval == "1wk":
        start_date_obj = end_date_obj - datetime.timedelta(days=1500)  # ~4 years for weekly
    else:
        start_date_obj = end_date_obj - datetime.timedelta(days=365)  # 1 year for daily

    # Convert to strings for yfinance
    start_str = start_date_obj.strftime("%Y-%m-%d")
    end_str = end_date_obj.strftime("%Y-%m-%d")

    # 3. Fetch using explicit start and end dates (NO 'period' argument!)
    df = yf.download(
        tickers_string,
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=True
    )

    # 4. Safely extract the Close prices regardless of yfinance version
    if isinstance(df.columns, pd.MultiIndex):
        # If yfinance returns a multi-level table (Price Type -> Tickers)
        if 'Close' in df.columns.get_level_values(0):
            df = df['Close']
        elif 'Adj Close' in df.columns.get_level_values(0):
            df = df['Adj Close']
    else:
        # If yfinance returns a single-level table
        if 'Close' in df.columns:
            # It's a single stock download
            df = df[['Close']]
            df.columns = tickers
        # If 'Close' isn't there, yfinance already made the columns the Tickers, so we do nothing!

    # If only 1 ticker is selected, force it to remain a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    # 5. Clean the data
    df = df.dropna(how='all')  # Drop rows where every stock is a NaN (weekends/holidays)
    df = df.ffill()  # Fill 1-day gaps
    df = df.dropna(axis=1)  # Drop columns that genuinely lack history

    return df