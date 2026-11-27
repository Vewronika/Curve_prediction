"""Run multiple prediction configurations and retain each outcome."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Callable, Iterable

import pandas as pd

from application.pipeline import PredictionRun, run_prediction
from data_acess import provider


@dataclass
class ExperimentConfig:
    """Inputs for one prediction inside an experiment."""

    data_mode: str
    k_prior: int
    k: int
    k_target: int | None
    custom_prices: pd.DataFrame | None = None
    all_tickers: Iterable[str] | None = None
    selected_tickers: Iterable[str] | None = None
    target_date: Any = None
    interval: str | None = None
    replace_incomplete: bool = True

    def to_prediction_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments accepted by the single-run pipeline."""
        return {
            "data_mode": self.data_mode,
            "k_prior": self.k_prior,
            "k": self.k,
            "k_target": self.k_target,
            "custom_prices": self.custom_prices,
            "all_tickers": self.all_tickers,
            "selected_tickers": self.selected_tickers,
            "target_date": self.target_date,
            "interval": self.interval,
            "replace_incomplete": self.replace_incomplete,
        }


@dataclass
class ExperimentRunResult:
    """Structured outcome for one configured prediction."""

    run_number: int
    config: ExperimentConfig
    succeeded: bool
    prediction: PredictionRun | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class ExperimentResult:
    """All outcomes from an experiment, including failed configurations."""

    runs: list[ExperimentRunResult]

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def successful_runs(self) -> int:
        return sum(run.succeeded for run in self.runs)

    @property
    def failed_runs(self) -> int:
        return self.total_runs - self.successful_runs


def run_experiment(
    configurations: Iterable[ExperimentConfig],
    *,
    prediction_runner: Callable[..., PredictionRun] | None = None,
) -> ExperimentResult:
    """Run every configuration, recording exceptions and continuing afterward.

    ``prediction_runner`` is injectable so tests can simulate successes and
    failures without making network calls or running the stochastic algorithm.
    """
    runner = prediction_runner or run_prediction
    outcomes: list[ExperimentRunResult] = []

    for run_number, config in enumerate(configurations, start=1):
        try:
            prediction = runner(**config.to_prediction_kwargs())
        except Exception as exc:
            outcomes.append(
                ExperimentRunResult(
                    run_number=run_number,
                    config=config,
                    succeeded=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        else:
            outcomes.append(
                ExperimentRunResult(
                    run_number=run_number,
                    config=config,
                    succeeded=True,
                    prediction=prediction,
                )
            )

    return ExperimentResult(runs=outcomes)


def build_k_target_configurations(
    base_config: ExperimentConfig,
    k_targets: Iterable[int],
) -> list[ExperimentConfig]:
    """Create one configuration per threshold, holding all other inputs fixed."""
    thresholds = list(k_targets)
    if not thresholds:
        raise ValueError("Choose at least one k_target value for the experiment.")

    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 3
        for value in thresholds
    ):
        raise ValueError("Every k_target must be a whole number of at least 3.")

    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Each k_target value must be unique in the experiment.")

    if base_config.data_mode == "csv" and base_config.custom_prices is not None:
        asset_count = base_config.custom_prices.shape[1]
        if any(value > asset_count for value in thresholds):
            raise ValueError(
                f"CSV data contains {asset_count} assets; k_target cannot exceed that count."
            )

    return [replace(base_config, k_target=int(value)) for value in thresholds]


def run_k_target_experiment(
    base_config: ExperimentConfig,
    k_targets: Iterable[int],
    *,
    prediction_runner: Callable[..., PredictionRun] | None = None,
    market_data_loader: Callable[..., tuple[pd.DataFrame, dict]] | None = None,
) -> ExperimentResult:
    """Run a threshold sweep while keeping the rest of the configuration fixed.

    A live market portfolio is loaded once and reused for each threshold. This
    keeps the comparison on the same observations and avoids repeated provider
    requests. Injected prediction runners are called directly for deterministic
    tests and callers that supply their own execution behavior.
    """
    configurations = build_k_target_configurations(base_config, k_targets)

    if prediction_runner is not None:
        return run_experiment(configurations, prediction_runner=prediction_runner)

    cached_market_data: tuple[pd.DataFrame, dict] | None = None
    market_data_attempted = False
    market_data_error: Exception | None = None

    def load_market_data_once(**kwargs) -> tuple[pd.DataFrame, dict]:
        nonlocal cached_market_data, market_data_attempted, market_data_error
        if not market_data_attempted:
            market_data_attempted = True
            loader = market_data_loader or provider.get_sliced_portfolio
            try:
                prices, report = loader(**kwargs)
            except Exception as exc:
                market_data_error = exc
            else:
                cached_market_data = (prices.copy(), deepcopy(report))
        if market_data_error is not None:
            raise market_data_error
        if cached_market_data is None:
            raise RuntimeError("Market data loading did not produce a result.")
        prices, report = cached_market_data
        return prices.copy(), deepcopy(report)

    def run_with_shared_market_data(**kwargs) -> PredictionRun:
        if kwargs["data_mode"] == "market":
            kwargs["market_data_loader"] = load_market_data_once
        return run_prediction(**kwargs)

    return run_experiment(configurations, prediction_runner=run_with_shared_market_data)
