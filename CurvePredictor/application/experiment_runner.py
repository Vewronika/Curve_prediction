"""Run multiple prediction configurations and retain each outcome."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
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
    b_percent: float = -10.0
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
            "b_percent": self.b_percent,
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


def build_portfolio_size_configurations(
    base_config: ExperimentConfig,
    portfolio_sizes: Iterable[int],
) -> list[ExperimentConfig]:
    """Create nested portfolios of different sizes, keeping k_target fixed."""
    sizes = list(portfolio_sizes)
    if not sizes:
        raise ValueError("Choose at least one portfolio size for the experiment.")
    if any(
        isinstance(size, bool)
        or not isinstance(size, Integral)
        or size < 3
        for size in sizes
    ):
        raise ValueError("Every portfolio size must be a whole number of at least 3.")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Each portfolio size must be unique in the experiment.")

    k_target = base_config.k_target
    if (
        isinstance(k_target, bool)
        or not isinstance(k_target, Integral)
        or k_target < 3
    ):
        raise ValueError("Choose a whole-number k_target of at least 3 before the experiment.")
    if any(size < k_target for size in sizes):
        raise ValueError("Every portfolio size must be at least as large as k_target.")

    if base_config.data_mode == "csv":
        if base_config.custom_prices is None:
            raise ValueError("Provide CSV price data before choosing portfolio sizes.")
        max_size = base_config.custom_prices.shape[1]
        if any(size > max_size for size in sizes):
            raise ValueError(
                f"CSV data contains {max_size} assets; portfolio size cannot exceed that count."
            )
        return [
            replace(
                base_config,
                custom_prices=base_config.custom_prices.iloc[:, :size].copy(),
            )
            for size in sizes
        ]

    if base_config.data_mode == "market":
        if base_config.selected_tickers is None:
            raise ValueError("Select market assets before choosing portfolio sizes.")
        selected_tickers = list(dict.fromkeys(base_config.selected_tickers))
        if not selected_tickers:
            raise ValueError("Select at least one market asset before choosing portfolio sizes.")
        if any(size > len(selected_tickers) for size in sizes):
            raise ValueError(
                f"Only {len(selected_tickers)} assets are selected; portfolio size cannot exceed that count."
            )

        all_tickers = (
            list(base_config.all_tickers) if base_config.all_tickers is not None else None
        )
        return [
            replace(
                base_config,
                all_tickers=all_tickers,
                selected_tickers=selected_tickers[:size],
            )
            for size in sizes
        ]

    raise ValueError(f"Unsupported data mode: {base_config.data_mode}")


def run_portfolio_size_experiment(
    base_config: ExperimentConfig,
    portfolio_sizes: Iterable[int],
    *,
    prediction_runner: Callable[..., PredictionRun] | None = None,
    market_data_loader: Callable[..., tuple[pd.DataFrame, dict]] | None = None,
) -> ExperimentResult:
    """Run an N sweep while holding the user-selected k_target fixed."""
    configurations = build_portfolio_size_configurations(base_config, portfolio_sizes)
    if base_config.data_mode != "market" or prediction_runner is not None:
        return run_experiment(configurations, prediction_runner=prediction_runner)

    largest_config = max(configurations, key=lambda config: len(config.selected_tickers))
    cached_prices: pd.DataFrame | None = None
    cached_report: dict | None = None
    market_data_attempted = False
    market_data_error: Exception | None = None

    def load_shared_portfolio(**kwargs) -> tuple[pd.DataFrame, dict]:
        nonlocal cached_prices, cached_report, market_data_attempted, market_data_error
        if not market_data_attempted:
            market_data_attempted = True
            loader = market_data_loader or provider.get_sliced_portfolio
            try:
                cached_prices, cached_report = loader(
                    all_tickers=kwargs["all_tickers"],
                    selected_tickers=largest_config.selected_tickers,
                    end_date=kwargs["end_date"],
                    interval=kwargs["interval"],
                    required_rows=kwargs["required_rows"],
                    replace_incomplete=kwargs["replace_incomplete"],
                )
            except Exception as exc:
                market_data_error = exc

        if market_data_error is not None:
            raise market_data_error
        if cached_prices is None or cached_report is None:
            raise RuntimeError("Market data loading did not produce a result.")

        requested_tickers = list(kwargs["selected_tickers"])
        eligible_selected = [
            ticker for ticker in requested_tickers if ticker in cached_prices.columns
        ]
        replacements = [
            ticker
            for ticker in cached_report.get("replacements", [])
            if ticker in cached_prices.columns
        ]
        tickers = eligible_selected + replacements[: len(requested_tickers) - len(eligible_selected)]
        prices = cached_prices.loc[:, tickers].copy()

        report = deepcopy(cached_report)
        report["requested_tickers"] = requested_tickers
        report["requested_count"] = len(requested_tickers)
        report["final_tickers"] = tickers
        report["final_count"] = len(tickers)
        report["shortfall"] = max(0, len(requested_tickers) - len(tickers))
        report["replacements"] = [ticker for ticker in tickers if ticker not in requested_tickers]
        report["removed"] = [
            removal
            for removal in cached_report.get("removed", [])
            if removal.get("ticker") in requested_tickers
            and not removal.get("replacement_candidate")
        ]
        report["observations"] = len(prices)
        return prices, report

    def run_with_shared_portfolio(**kwargs) -> PredictionRun:
        kwargs["market_data_loader"] = load_shared_portfolio
        return run_prediction(**kwargs)

    return run_experiment(configurations, prediction_runner=run_with_shared_portfolio)


def build_k_configurations(
    base_config: ExperimentConfig,
    k_values: Iterable[int],
) -> list[ExperimentConfig]:
    """Create one configuration per K, keeping K_prior and other inputs fixed."""
    windows = list(k_values)
    if not windows:
        raise ValueError("Choose at least one K value for the experiment.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
        for value in windows
    ):
        raise ValueError("Every K value must be a whole number of at least 1.")
    if len(set(windows)) != len(windows):
        raise ValueError("Each K value must be unique in the experiment.")
    if (
        isinstance(base_config.k_prior, bool)
        or not isinstance(base_config.k_prior, Integral)
        or base_config.k_prior < 1
    ):
        raise ValueError("Choose a fixed K_prior of at least 1 before the experiment.")

    if base_config.data_mode == "csv":
        if base_config.custom_prices is None:
            raise ValueError("Provide CSV price data before choosing K values.")
        max_window = len(base_config.custom_prices) - int(base_config.k_prior) - 1
        if any(value > max_window for value in windows):
            raise ValueError(
                f"CSV data supports a maximum K of {max_window} "
                f"with K_prior={base_config.k_prior}."
            )
    elif base_config.data_mode == "market":
        if base_config.selected_tickers is None:
            raise ValueError("Select market assets before choosing K values.")
        selected_tickers = list(dict.fromkeys(base_config.selected_tickers))
        if not selected_tickers:
            raise ValueError("Select market assets before choosing K values.")
        all_tickers = (
            list(base_config.all_tickers) if base_config.all_tickers is not None else None
        )
        base_config = replace(
            base_config,
            all_tickers=all_tickers,
            selected_tickers=selected_tickers,
        )
    else:
        raise ValueError(f"Unsupported data mode: {base_config.data_mode}")

    return [replace(base_config, k=int(value)) for value in windows]


def run_k_experiment(
    base_config: ExperimentConfig,
    k_values: Iterable[int],
    *,
    prediction_runner: Callable[..., PredictionRun] | None = None,
    market_data_loader: Callable[..., tuple[pd.DataFrame, dict]] | None = None,
) -> ExperimentResult:
    """Run a K sweep, loading enough live history once for the largest K."""
    configurations = build_k_configurations(base_config, k_values)
    if base_config.data_mode != "market" or prediction_runner is not None:
        return run_experiment(configurations, prediction_runner=prediction_runner)

    largest_config = max(configurations, key=lambda config: config.k)
    largest_required_rows = largest_config.k_prior + largest_config.k + 1
    cached_market_data: tuple[pd.DataFrame, dict] | None = None
    market_data_attempted = False
    market_data_error: Exception | None = None

    def load_market_data_once(**kwargs) -> tuple[pd.DataFrame, dict]:
        nonlocal cached_market_data, market_data_attempted, market_data_error
        if not market_data_attempted:
            market_data_attempted = True
            loader = market_data_loader or provider.get_sliced_portfolio
            try:
                prices, report = loader(
                    all_tickers=kwargs["all_tickers"],
                    selected_tickers=kwargs["selected_tickers"],
                    end_date=kwargs["end_date"],
                    interval=kwargs["interval"],
                    required_rows=largest_config.k_prior + largest_config.k + 1,
                    replace_incomplete=kwargs["replace_incomplete"],
                )
            except Exception as exc:
                market_data_error = exc
            else:
                cached_market_data = (prices.copy(), deepcopy(report))
        if market_data_error is not None:
            if kwargs["required_rows"] < largest_required_rows:
                loader = market_data_loader or provider.get_sliced_portfolio
                prices, report = loader(**kwargs)
                return prices.copy(), deepcopy(report)
            raise market_data_error
        if cached_market_data is None:
            raise RuntimeError("Market data loading did not produce a result.")
        prices, report = cached_market_data
        return prices.copy(), deepcopy(report)

    def run_with_shared_market_data(**kwargs) -> PredictionRun:
        kwargs["market_data_loader"] = load_market_data_once
        return run_prediction(**kwargs)

    return run_experiment(configurations, prediction_runner=run_with_shared_market_data)


def build_b_configurations(
    base_config: ExperimentConfig,
    b_percent_values: Iterable[float],
) -> list[ExperimentConfig]:
    """Create one configuration per percent-scale B value, holding other inputs fixed."""
    values = list(b_percent_values)
    if not values:
        raise ValueError("Choose at least one B value for the experiment.")

    try:
        percentages = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Every B value must be a number between -100 and 0 percent.") from exc
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(percent)
        or not -100 <= percent <= 0
        for value, percent in zip(values, percentages)
    ):
        raise ValueError("Every B value must be between -100 and 0 percent.")
    if len(set(percentages)) != len(percentages):
        raise ValueError("Each B value must be unique in the experiment.")

    if base_config.data_mode == "market":
        if base_config.selected_tickers is None:
            raise ValueError("Select market assets before varying B.")
        selected_tickers = list(dict.fromkeys(base_config.selected_tickers))
        if not selected_tickers:
            raise ValueError("Select market assets before varying B.")
        all_tickers = (
            list(base_config.all_tickers) if base_config.all_tickers is not None else None
        )
        base_config = replace(
            base_config,
            all_tickers=all_tickers,
            selected_tickers=selected_tickers,
        )
    elif base_config.data_mode == "csv" and base_config.custom_prices is None:
        raise ValueError("Provide CSV price data before varying B.")
    elif base_config.data_mode != "csv":
        raise ValueError(f"Unsupported data mode: {base_config.data_mode}")

    return [replace(base_config, b_percent=value) for value in percentages]


def build_b_percent_values(
    start: int = -50,
    end: int = 0,
    step: int = 2,
) -> list[int]:
    """Build the default B sweep in whole percent values, including its endpoint."""
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (start, end, step)
    ):
        raise ValueError("B range endpoints and step must be whole numbers.")
    if not (-100 <= start <= 0) or not (-100 <= end <= 0):
        raise ValueError("B range endpoints must be between -100 and 0 percent.")
    if start > end:
        raise ValueError("B range start must be less than or equal to its end.")
    if step < 1:
        raise ValueError("B step must be at least 1 percent.")

    values = list(range(start, end + 1, step))
    if values[-1] != end:
        values.append(end)
    return values


def run_b_experiment(
    base_config: ExperimentConfig,
    b_percent_values: Iterable[float],
    *,
    prediction_runner: Callable[..., PredictionRun] | None = None,
    market_data_loader: Callable[..., tuple[pd.DataFrame, dict]] | None = None,
) -> ExperimentResult:
    """Run a B sweep, reusing one live-market dataset for every B value."""
    configurations = build_b_configurations(base_config, b_percent_values)
    if base_config.data_mode != "market" or prediction_runner is not None:
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
        kwargs["market_data_loader"] = load_market_data_once
        return run_prediction(**kwargs)

    return run_experiment(configurations, prediction_runner=run_with_shared_market_data)
