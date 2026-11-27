import unittest
from unittest.mock import Mock, patch

import pandas as pd

from application.experiment_runner import (
    ExperimentConfig,
    build_k_target_configurations,
    run_experiment,
    run_k_target_experiment,
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


class KTargetExperimentTests(unittest.TestCase):
    def test_builder_varies_only_k_target(self):
        base = sample_config(k_target=3)
        base.custom_prices["DDD"] = [40.0, 41.0]
        base.custom_prices["EEE"] = [50.0, 51.0]

        configurations = build_k_target_configurations(base, [3, 4, 5])

        self.assertEqual([config.k_target for config in configurations], [3, 4, 5])
        for config in configurations:
            self.assertEqual(config.data_mode, base.data_mode)
            self.assertEqual(config.k_prior, base.k_prior)
            self.assertEqual(config.k, base.k)
            self.assertIs(config.custom_prices, base.custom_prices)

    def test_builder_rejects_empty_duplicate_and_invalid_thresholds(self):
        base = sample_config()

        for values, message in [
            ([], "at least one"),
            ([3, 3], "unique"),
            ([2], "at least 3"),
            ([3.5], "whole number"),
            ([True], "whole number"),
            ([4], "cannot exceed"),
        ]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    build_k_target_configurations(base, values)

    def test_sweep_runs_each_threshold_and_preserves_partial_failures(self):
        base = sample_config()
        base.custom_prices["DDD"] = [40.0, 41.0]
        successful_prediction = sample_prediction(k_target=4)
        runner = Mock(side_effect=[ValueError("threshold failed"), successful_prediction])

        experiment = run_k_target_experiment(base, [3, 4], prediction_runner=runner)

        self.assertEqual(experiment.total_runs, 2)
        self.assertEqual(experiment.failed_runs, 1)
        self.assertEqual(experiment.successful_runs, 1)
        self.assertEqual([run.config.k_target for run in experiment.runs], [3, 4])
        self.assertEqual(experiment.runs[0].error_message, "threshold failed")
        self.assertIs(experiment.runs[1].prediction, successful_prediction)
        self.assertEqual(runner.call_args_list[0].kwargs["k_target"], 3)
        self.assertEqual(runner.call_args_list[1].kwargs["k_target"], 4)

    def test_live_market_data_is_loaded_once_for_all_thresholds(self):
        base = ExperimentConfig(
            data_mode="market",
            k_prior=1,
            k=1,
            k_target=3,
            all_tickers=["AAA", "BBB", "CCC"],
            selected_tickers=["AAA", "BBB", "CCC"],
            target_date="2026-01-06",
            interval="1d",
        )
        prices = sample_config().custom_prices
        loader = Mock(return_value=(prices, {"provider": "fixture"}))
        calls = []

        def fake_prediction_runner(**kwargs):
            loaded_prices, _ = kwargs["market_data_loader"](
                all_tickers=kwargs["all_tickers"],
                selected_tickers=kwargs["selected_tickers"],
                end_date=kwargs["target_date"],
                interval=kwargs["interval"],
                required_rows=kwargs["k_prior"] + kwargs["k"] + 1,
                replace_incomplete=kwargs["replace_incomplete"],
            )
            calls.append((kwargs["k_target"], loaded_prices))
            return sample_prediction(k_target=kwargs["k_target"])

        # Exercise the production wrapper around run_prediction while replacing
        # the model pipeline itself, so no market request or algorithm runs.
        def prediction_side_effect(**kwargs):
            return fake_prediction_runner(**kwargs)

        with patch("application.experiment_runner.run_prediction", side_effect=prediction_side_effect):
            experiment = run_k_target_experiment(
                base,
                [3, 4],
                market_data_loader=loader,
            )

        self.assertEqual(experiment.successful_runs, 2)
        loader.assert_called_once()
        self.assertEqual([threshold for threshold, _ in calls], [3, 4])
        self.assertTrue(calls[0][1].equals(calls[1][1]))
        self.assertIsNot(calls[0][1], calls[1][1])


if __name__ == "__main__":
    unittest.main()
