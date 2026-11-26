"""Run multiple prediction configurations and retain each outcome."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pandas as pd

from application.pipeline import PredictionRun, run_prediction


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
