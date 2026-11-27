"""Reusable orchestration for one CurvePredictor analysis run."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Iterable

import pandas as pd

from data_acess import math_adapter, provider
from method import algorithm


class InsufficientObservationsError(ValueError):
    """Raised when an input dataset is shorter than the configured windows."""


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
    b_percent: float = -10.0


def run_prediction(
    *,
    data_mode: str,
    k_prior: int,
    k: int,
    k_target: int | None,
    b_percent: float = -10.0,
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
    k_prior = _window_steps("K_prior", k_prior)
    k = _window_steps("K", k)
    if k_prior < 1 or k < 1:
        raise ValueError("Both time windows must be at least 1 step.")
    b_percent = _barrier_percent(b_percent)
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

    available_rows = len(prices.dropna(axis=0, how="any"))
    if available_rows < required_rows:
        if data_mode == "market":
            raise provider.InsufficientHistoricalDataError(
                f"Insufficient historical data: requested {required_rows} observations, "
                f"but only {available_rows} complete observations are available "
                f"for interval '{interval}'."
            )
        raise InsufficientObservationsError(
            f"Insufficient CSV data: requested {required_rows} observations, "
            f"but only {available_rows} complete observations are available."
        )

    if prices.shape[1] < 3:
        raise ValueError(
            f"Only {prices.shape[1]} usable assets are available. At least 3 are required."
        )
    try:
        integer_k_target = int(k_target)
    except (TypeError, ValueError, OverflowError):
        integer_k_target = None
    if (
        isinstance(k_target, bool)
        or integer_k_target is None
        or integer_k_target != k_target
        or not (3 <= integer_k_target <= prices.shape[1])
    ):
        raise ValueError(
            f"Crash threshold k={k_target} is invalid for the final portfolio size "
            f"N={prices.shape[1]}. Choose a threshold from 3 to N."
        )
    k_target = integer_k_target

    prior_inputs, active_inputs, asset_count = math_adapter.df_to_math_inputs(
        prices, k_prior, k
    )
    if asset_count != prices.shape[1]:
        raise ValueError("The asset count does not match the input data.")

    runner = model_runner or algorithm.run_algorithm
    model_output = runner(
        prior_inputs,
        active_inputs,
        asset_count,
        k_prior,
        k,
        k_target,
        B=b_percent / 100.0,
    )

    return PredictionRun(
        prices=prices,
        data_report=data_report,
        model_output=model_output,
        asset_count=asset_count,
        k_prior=k_prior,
        k=k,
        k_target=k_target,
        b_percent=b_percent,
    )


def _window_steps(name: str, value: int) -> int:
    """Accept integer step counts without silently truncating decimal inputs."""
    try:
        steps = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a whole number of steps.") from exc
    if isinstance(value, bool) or steps != value:
        raise ValueError(f"{name} must be a whole number of steps.")
    return steps


def _barrier_percent(value: int | float) -> float:
    """Validate the user's percent-scale barrier and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("B must be a number between -100 and 0 percent.")
    try:
        percent = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("B must be a number between -100 and 0 percent.") from exc
    if not math.isfinite(percent) or not -100 <= percent <= 0:
        raise ValueError("B must be between -100 and 0 percent.")
    return percent
