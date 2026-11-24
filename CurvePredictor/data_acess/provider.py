"""Market-data access and validation for the CurvePredictor app.

Prices are never forward/back-filled. For a live portfolio, incomplete assets can
be excluded and optionally replaced by eligible assets from the same universe.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:  # Keep validation helpers unit-testable without the UI package.
    class _StreamlitCacheFallback:
        @staticmethod
        def cache_data(*args, **kwargs):
            def decorate(function):
                return function
            return decorate
    st = _StreamlitCacheFallback()


class InsufficientHistoricalDataError(ValueError):
    """Raised when market data exists but cannot satisfy the requested window."""


class NoMarketDataError(ValueError):
    """Raised when the market-data provider returns no rows at all."""



def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide CSV: first column is dates; remaining columns are prices."""
    if df.empty:
        raise ValueError("The CSV contains no data rows.")
    if len(df.columns) < 2:
        raise ValueError("The CSV must contain a date column and at least one asset column.")
    if not df.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names.")

    date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Column '{date_col}' contains missing or invalid dates.")
    if dates.duplicated().any():
        raise ValueError("The CSV contains duplicate dates.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be in chronological order, from oldest to newest.")

    prices = df.iloc[:, 1:].copy()
    prices.columns = prices.columns.astype(str).str.strip()
    if any(not c for c in prices.columns):
        raise ValueError("Asset column names cannot be empty.")
    if not prices.columns.is_unique:
        raise ValueError("Asset column names must be unique after trimming whitespace.")

    for col in prices.columns:
        values = pd.to_numeric(prices[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Asset '{col}' contains missing or non-numeric prices.")
        if (values <= 0).any():
            raise ValueError(f"Asset '{col}' contains zero or negative prices.")
        prices[col] = values.astype(float)

    prices.index = pd.DatetimeIndex(dates, name="Date")
    return prices


def parse_uploaded_csv(csv_file) -> pd.DataFrame:
    try:
        frame = pd.read_csv(csv_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")
    return validate_dataframe(frame)


def _date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("Target date must be a valid date.") from exc


def _period_days(interval: str, required_rows: int) -> int:
    """Estimate calendar span needed for requested observations, with buffer."""
    if interval == "1d":
        return max(30, int(required_rows * 1.7) + 10)
    if interval == "1wk":
        return max(90, int(required_rows * 8) + 30)
    if interval == "1h":
        # Yahoo generally limits intraday history; the caller will get a clear
        # insufficient-history error if the requested window exceeds availability.
        # The request's end is exclusive and includes end_date + 1 day, so cap
        # the lookback at 728 to keep the full requested range below 730 days.
        return min(728, max(7, int(required_rows * 0.3) + 7))
    raise ValueError(f"Unsupported data frequency: {interval}")


@st.cache_data(ttl=900, max_entries=20, show_spinner=False)
def fetch_stock_data(tickers: list[str], end_date, interval: str = "1d", required_rows: int = 61) -> pd.DataFrame:
    """Fetch adjusted close prices. Returns incomplete columns without filling them."""
    tickers = list(dict.fromkeys(str(t).strip() for t in tickers if str(t).strip()))
    if not tickers:
        raise ValueError("No companies were provided.")
    end = _date(end_date)
    start = end - dt.timedelta(days=_period_days(interval, required_rows))

    try:
        import yfinance as yf
        downloaded = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            # yfinance's end date is exclusive; include the selected anchor date.
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:
        raise ValueError(f"Market-data download failed: {exc}") from exc

    if downloaded is None or downloaded.empty:
        raise NoMarketDataError(
            f"The market-data provider returned no data for interval '{interval}' "
            f"from {start.isoformat()} through {end.isoformat()}. "
            "The provider may not offer data for that date range or interval."
        )

    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        price_field = "Close" if "Close" in level0 else "Adj Close" if "Adj Close" in level0 else None
        if price_field is None:
            raise ValueError("Downloaded market data has no closing-price field.")
        prices = downloaded[price_field].copy()
    else:
        field = "Close" if "Close" in downloaded.columns else "Adj Close" if "Adj Close" in downloaded.columns else None
        if field is None:
            raise ValueError("Downloaded market data has no closing-price field.")
        prices = downloaded[[field]].copy()
        if len(tickers) != 1:
            raise ValueError("Unexpected market-data format for a multi-company request.")
        prices.columns = tickers

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")

    # yfinance can return fewer columns than requested; preserve requested order.
    for ticker in tickers:
        if ticker not in prices.columns:
            prices[ticker] = np.nan
    prices = prices.reindex(columns=tickers)
    return prices


def assess_price_matrix(prices: pd.DataFrame, required_rows: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return complete eligible assets and a per-asset exclusion reason."""
    if prices.empty:
        raise ValueError("No market prices were returned for the requested period.")
    reasons: dict[str, str] = {}
    eligible = []
    for ticker in prices.columns:
        series = pd.to_numeric(prices[ticker], errors="coerce")
        if series.notna().sum() == 0:
            reasons[ticker] = "no price history returned"
        elif series.isna().any():
            reasons[ticker] = f"{int(series.isna().sum())} missing observations (not imputed)"
        elif not np.isfinite(series.to_numpy(dtype=float)).all():
            reasons[ticker] = "non-finite price values"
        elif (series <= 0).any():
            reasons[ticker] = "zero or negative prices"
        elif len(series) < required_rows:
            reasons[ticker] = f"only {len(series)} observations; {required_rows} required"
        else:
            eligible.append(ticker)
    return prices[eligible].copy(), reasons


def get_sliced_portfolio(
    all_tickers: Iterable[str],
    selected_tickers: Iterable[str],
    end_date,
    interval: str = "1d",
    required_rows: int = 61,
    replace_incomplete: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Fetch selected assets, optionally replacing incomplete ones from same universe.

    Replacements are attempted in the supplied universe order. Every removal and
    replacement is returned in a report; missing observations are never filled.
    """
    universe = list(dict.fromkeys(str(t).strip() for t in all_tickers if str(t).strip()))
    selected = list(dict.fromkeys(str(t).strip() for t in selected_tickers if str(t).strip()))
    if not selected:
        raise ValueError("Select at least one company.")
    unknown = [t for t in selected if t not in set(universe)]
    if unknown:
        raise ValueError("Selected assets are not in the chosen universe: " + ", ".join(unknown))

    initial = fetch_stock_data(selected, end_date, interval, required_rows)
    if len(initial) < required_rows:
        available_rows = len(initial.dropna(axis=0, how="any"))
        raise InsufficientHistoricalDataError(
            f"Insufficient historical data: requested {required_rows} observations, "
            f"but only {available_rows} complete observations are available for "
            f"interval '{interval}'."
        )
    eligible, reasons = assess_price_matrix(initial, required_rows)
    kept = list(eligible.columns)
    removed = [{"ticker": ticker, "reason": reasons[ticker]} for ticker in selected if ticker in reasons]
    replacements: list[str] = []

    if replace_incomplete and removed:
        candidates = [t for t in universe if t not in set(selected)]
        if candidates:
            # Fetch in modest batches to avoid a single very large provider request.
            for offset in range(0, len(candidates), 40):
                batch = candidates[offset:offset + 40]
                candidate_prices = fetch_stock_data(batch, end_date, interval, required_rows)
                candidate_eligible, candidate_reasons = assess_price_matrix(candidate_prices, required_rows)
                for ticker in batch:
                    if len(kept) >= len(selected):
                        break
                    if ticker in candidate_eligible.columns:
                        kept.append(ticker)
                        replacements.append(ticker)
                    elif ticker in candidate_reasons:
                        # Keep a bounded record of rejected candidates for transparency.
                        if len(removed) < len(selected) + 10:
                            removed.append({"ticker": ticker, "reason": candidate_reasons[ticker], "replacement_candidate": True})
                if len(kept) >= len(selected):
                    break

    if not kept:
        raise ValueError("No selected assets have a complete, usable price history for this period.")

    # Use one common set of real timestamps. No imputation: if the selected
    # assets don't share complete rows, exclude rows that are incomplete across
    # the final portfolio, then enforce required length.
    final = fetch_stock_data(kept, end_date, interval, required_rows)
    final = final.dropna(axis=0, how="any")
    final = final.loc[np.isfinite(final.to_numpy(dtype=float)).all(axis=1)]
    final = final.loc[(final > 0).all(axis=1)]
    if len(final) < required_rows:
        raise InsufficientHistoricalDataError(
            f"Insufficient historical data: requested {required_rows} observations, "
            f"but only {len(final)} complete observations are available for "
            f"interval '{interval}'. No prices were filled. Try a shorter window."
        )

    final_n = len(final.columns)
    report = {
        "requested_tickers": selected,
        "requested_count": len(selected),
        "removed": removed,
        "replacements": replacements,
        "final_tickers": list(final.columns),
        "final_count": final_n,
        "shortfall": max(0, len(selected) - final_n),
        "observations": len(final),
        "replace_incomplete": replace_incomplete,
        "provider": "Yahoo Finance via yfinance",
        "note": "Membership reflects the database's currently stored universe, not point-in-time historical membership.",
    }
    return final, report
