import unittest
from unittest.mock import Mock

import pandas as pd

from application.experiment_runner import (
    ExperimentConfig,
    run_experiment,
)
from application.pipeline import PredictionRun


def sample_config(k_target=3):
    prices = pd.DataFrame(
        {"AAA": [10.0, 11.0], "BBB": [20.0, 21.0], "CCC": [30.0, 31.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )
    return ExperimentConfig(
        data_mode="csv",
        custom_prices=prices,
        k_prior=1,
        k=1,
        k_target=k_target,
    )


def sample_prediction(k_target=3):
    return PredictionRun(
        prices=sample_config().custom_prices,
        data_report={"provider": "User-uploaded CSV"},
        model_output={"crash_probability": 0.25},
        asset_count=3,
        k_prior=1,
        k=1,
        k_target=k_target,
    )


class RunExperimentTests(unittest.TestCase):
    def test_successful_configuration_returns_structured_prediction(self):
        config = sample_config(k_target=3)
        prediction = sample_prediction(k_target=3)
        runner = Mock(return_value=prediction)

        experiment = run_experiment([config], prediction_runner=runner)

        self.assertEqual(experiment.total_runs, 1)
        self.assertEqual(experiment.successful_runs, 1)
        self.assertEqual(experiment.failed_runs, 0)
        self.assertTrue(experiment.runs[0].succeeded)
        self.assertIs(experiment.runs[0].prediction, prediction)
        self.assertIsNone(experiment.runs[0].error_message)
        runner.assert_called_once_with(**config.to_prediction_kwargs())

    def test_failed_configuration_is_recorded_and_next_run_still_executes(self):
        first_config = sample_config(k_target=2)
        second_config = sample_config(k_target=3)
        second_prediction = sample_prediction(k_target=3)
        runner = Mock(side_effect=[ValueError("invalid threshold"), second_prediction])

        experiment = run_experiment(
            [first_config, second_config],
            prediction_runner=runner,
        )

        self.assertEqual(experiment.total_runs, 2)
        self.assertEqual(experiment.successful_runs, 1)
        self.assertEqual(experiment.failed_runs, 1)
        self.assertFalse(experiment.runs[0].succeeded)
        self.assertEqual(experiment.runs[0].error_type, "ValueError")
        self.assertEqual(experiment.runs[0].error_message, "invalid threshold")
        self.assertEqual(experiment.runs[0].run_number, 1)
        self.assertTrue(experiment.runs[1].succeeded)
        self.assertIs(experiment.runs[1].prediction, second_prediction)
        self.assertEqual(experiment.runs[1].run_number, 2)
        self.assertEqual(runner.call_count, 2)

    def test_empty_experiment_returns_zero_counts(self):
        experiment = run_experiment([], prediction_runner=Mock())

        self.assertEqual(experiment.total_runs, 0)
        self.assertEqual(experiment.successful_runs, 0)
        self.assertEqual(experiment.failed_runs, 0)
        self.assertEqual(experiment.runs, [])


if __name__ == "__main__":
    unittest.main()
