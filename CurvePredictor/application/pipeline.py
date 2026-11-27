"""Reusable orchestration for one CurvePredictor analysis run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pandas as pd

from data_acess import math_adapter, provider
from method import algorithm


@dataclass
class PredictionRun:
    """Inputs and outputs needed by the UI to present a completed run."""

    prices: pd.DataFrame
    data_report: dict[str, Any]
    model_output: dict[str, Any]
    asset_count: int
    k_prior: int
    k: int
    k_target: int


def run_prediction(
    *,
    data_mode: str,
    k_prior: int,
    k: int,
    k_target: int | None,
    custom_prices: pd.DataFrame | None = None,
    all_tickers: Iterable[str] | None = None,
    selected_tickers: Iterable[str] | None = None,
    target_date=None,
    interval: str | None = None,
    replace_incomplete: bool = True,
    market_data_loader: Callable[..., tuple[pd.DataFrame, dict]] | None = None,
    model_runner: Callable[..., dict] | None = None,
) -> PredictionRun:
    """Load, validate, adapt, and execute a single model run.

    Dependencies can be supplied by callers for deterministic use. The Streamlit
    page only gathers inputs and renders the returned data; this function owns
    the run's orchestration and validation.
    """
    k_prior = int(k_prior)
    k = int(k)
    if k_prior < 1 or k < 1:
        raise ValueError("Both time windows must be at least 1 step.")
    required_rows = k_prior + k + 1

    if data_mode == "csv":
        if custom_prices is None:
            raise ValueError("Upload a valid CSV before running the analysis.")
        prices = custom_prices.copy()
        data_report = {
            "requested_count": prices.shape[1],
            "final_count": prices.shape[1],
            "requested_tickers": list(prices.columns),
            "final_tickers": list(prices.columns),
            "removed": [],
            "replacements": [],
            "shortfall": 0,
            "observations": len(prices),
            "provider": "User-uploaded CSV",
            "note": "CSV data supplied by the user; no market-universe membership metadata.",
        }
    elif data_mode == "market":
        if target_date is None or interval is None:
            raise ValueError("A target date and sampling interval are required for live market data.")
        if all_tickers is None or selected_tickers is None:
            raise ValueError("Select a market universe and at least one asset.")
        loader = market_data_loader or provider.get_sliced_portfolio
        prices, data_report = loader(
            all_tickers=all_tickers,
            selected_tickers=selected_tickers,
            end_date=target_date,
            interval=interval,
            required_rows=required_rows,
            replace_incomplete=replace_incomplete,
        )
    else:
        raise ValueError(f"Unsupported data mode: {data_mode}")

    if prices.shape[1] < 3:
        raise ValueError(
            f"Only {prices.shape[1]} usable assets are available. At least 3 are required."
        )
    if k_target is None or not (3 <= k_target <= prices.shape[1]):
        raise ValueError(
            f"Crash threshold k={k_target} is invalid for the final portfolio size "
            f"N={prices.shape[1]}. Choose a threshold from 3 to N."
        )

    prior_inputs, active_inputs, asset_count = math_adapter.df_to_math_inputs(
        prices, k_prior, k
    )
    if asset_count != prices.shape[1]:
        raise ValueError("The asset count does not match the input data.")

    runner = model_runner or algorithm.run_algorithm
    model_output = runner(prior_inputs, active_inputs, asset_count, k_prior, k, k_target)

    return PredictionRun(
        prices=prices,
        data_report=data_report,
        model_output=model_output,
        asset_count=asset_count,
        k_prior=k_prior,
        k=k,
        k_target=k_target,
    )
