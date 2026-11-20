import sqlite3
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime
from config import DB_PATH
import numpy as np


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide-format price matrix and return it indexed by Date."""
    if df.empty:
        raise ValueError("The CSV contains no data rows.")

    if len(df.columns) < 2:
        raise ValueError(
            "The CSV must contain a date column and at least one asset column."
        )

    if not df.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names.")

    # The first column is the date/time index
    date_column = df.columns[0]
    dates = pd.to_datetime(df[date_column], errors="coerce")

    if dates.isna().any():
        raise ValueError(
            f"Column '{date_column}' contains missing or invalid dates."
        )

    if dates.duplicated().any():
        raise ValueError("The CSV contains duplicate dates.")

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Dates must be in chronological order, from oldest to newest."
        )

    prices = df.iloc[:, 1:].copy()
    prices.columns = prices.columns.astype(str).str.strip()

    if any(not column for column in prices.columns):
        raise ValueError("Asset column names cannot be empty.")

    if not prices.columns.is_unique:
        raise ValueError(
            "Asset column names must be unique after removing surrounding spaces."
        )

    if prices.empty or prices.shape[1] == 0:
        raise ValueError("The CSV must contain at least one asset.")

    for column in prices.columns:
        original = prices[column]
        numeric = pd.to_numeric(original, errors="coerce")

        # Reject blanks, non-numeric cells, NaN and infinite values.
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(
                f"Asset '{column}' contains missing or non-numeric prices."
            )

        if (numeric <= 0).any():
            raise ValueError(
                f"Asset '{column}' contains zero or negative prices."
            )

        prices[column] = numeric

    prices.index = pd.DatetimeIndex(dates, name="Date")
    return prices


def parse_uploaded_csv(csv_file) -> pd.DataFrame:
    """Read, validate and normalize an uploaded wide-format CSV."""
    try:
        df = pd.read_csv(csv_file)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    return validate_dataframe(df)




def get_sliced_portfolio(all_tickers, selected_tickers, end_date, interval="1d"):
    """Fetch market data and return the selected assets only."""
    if not selected_tickers:
        raise ValueError("Select at least one company.")

    giant_df = fetch_stock_data(all_tickers, end_date, interval)

    missing_tickers = [
        ticker for ticker in selected_tickers
        if ticker not in giant_df.columns
    ]
    if missing_tickers:
        raise ValueError(
            "Market data is unavailable for: "
            + ", ".join(missing_tickers)
        )

    result = giant_df[selected_tickers].copy()

    if result.empty:
        raise ValueError("No market data is available for the selected companies.")

    return result


@st.cache_data(ttl="15m", max_entries=10)
def fetch_stock_data(tickers: list, end_date, interval: str = "1d") -> pd.DataFrame:
    """Download and validate historical adjusted closing prices."""
    if not tickers:
        raise ValueError("No companies were provided.")

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

    df = yf.download(
        " ".join(tickers),
        start=start_date_obj.strftime("%Y-%m-%d"),
        end=end_date_obj.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError("The market-data provider returned no data.")

    # Extract closing prices from yfinance's possible column formats.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        elif "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]
        else:
            raise ValueError("The downloaded data has no closing-price column.")
    elif "Close" in df.columns:
        df = df[["Close"]].copy()
        if len(tickers) == 1:
            df.columns = tickers
        else:
            raise ValueError("Unexpected market-data format for multiple companies.")
    elif "Adj Close" in df.columns:
        df = df[["Adj Close"]].copy()
        if len(tickers) == 1:
            df.columns = tickers
        else:
            raise ValueError("Unexpected market-data format for multiple companies.")
    else:
        raise ValueError("The downloaded data has no closing-price column.")

    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])

    # Remove dates with no price for any selected asset.
    df = df.dropna(how="all")
    # Remove assets for which the provider returned no prices at all.
    df = df.dropna(axis=1, how="all")

    if df.empty or df.shape[1] == 0:
        raise ValueError("No usable market prices were returned.")

    # Keep real provider dates; do not create weekend or holiday rows.
    df = df.sort_index()

    if not df.index.is_unique:
        raise ValueError("Market data contains duplicate timestamps.")

    # Do not silently fill missing observations.
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        details = ", ".join(
            f"{ticker}: {count} missing"
            for ticker, count in missing_counts.items()
        )
        raise ValueError(
            "Market data contains missing prices. No prices were filled. "
            f"Affected assets: {details}. Try a different date range or "
            "remove affected assets from the portfolio."
        )

    prices = df.to_numpy(dtype=float)

    if not np.isfinite(prices).all():
        raise ValueError("Market data contains infinite or non-finite prices.")

    if (prices <= 0).any():
        raise ValueError("Market data contains zero or negative prices.")

    return df
