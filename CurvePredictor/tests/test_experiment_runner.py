import unittest
from unittest.mock import Mock, patch

import pandas as pd

from application.experiment_runner import (
    ExperimentConfig,
    build_b_configurations,
    build_b_percent_values,
    build_k_configurations,
    build_k_target_configurations,
    build_portfolio_size_configurations,
    run_experiment,
    run_b_experiment,
    run_k_experiment,
    run_k_target_experiment,
    run_portfolio_size_experiment,
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


class PortfolioSizeExperimentTests(unittest.TestCase):
    def test_csv_builder_varies_n_and_holds_k_target_fixed(self):
        base = sample_config(k_target=3)
        base.custom_prices["DDD"] = [40.0, 41.0]
        base.custom_prices["EEE"] = [50.0, 51.0]

        configurations = build_portfolio_size_configurations(base, [3, 4, 5])

        self.assertEqual([config.custom_prices.shape[1] for config in configurations], [3, 4, 5])
        self.assertEqual([config.k_target for config in configurations], [3, 3, 3])
        self.assertEqual([config.k for config in configurations], [base.k] * 3)
        self.assertEqual([config.k_prior for config in configurations], [base.k_prior] * 3)
        self.assertEqual(
            [list(config.custom_prices.columns) for config in configurations],
            [list(base.custom_prices.columns[:size]) for size in [3, 4, 5]],
        )

    def test_market_builder_uses_nested_selected_tickers_and_same_universe(self):
        base = ExperimentConfig(
            data_mode="market",
            k_prior=2,
            k=4,
            k_target=3,
            all_tickers=["AAA", "BBB", "CCC", "DDD", "EEE"],
            selected_tickers=["AAA", "BBB", "CCC", "DDD"],
            target_date="2026-01-06",
            interval="1d",
        )

        configurations = build_portfolio_size_configurations(base, [3, 4])

        self.assertEqual(
            [config.selected_tickers for config in configurations],
            [["AAA", "BBB", "CCC"], ["AAA", "BBB", "CCC", "DDD"]],
        )
        self.assertEqual(configurations[0].all_tickers, configurations[1].all_tickers)
        self.assertEqual([config.k_target for config in configurations], [3, 3])

    def test_builder_rejects_sizes_that_cannot_use_fixed_k_target(self):
        base = sample_config(k_target=4)

        with self.assertRaisesRegex(ValueError, "at least as large as k_target"):
            build_portfolio_size_configurations(base, [3, 4])

    def test_runner_executes_each_portfolio_size(self):
        base = sample_config(k_target=3)
        base.custom_prices["DDD"] = [40.0, 41.0]
        runner = Mock(
            side_effect=[sample_prediction(k_target=3), sample_prediction(k_target=3)]
        )

        experiment = run_portfolio_size_experiment(
            base,
            [3, 4],
            prediction_runner=runner,
        )

        self.assertEqual(experiment.total_runs, 2)
        self.assertEqual(experiment.successful_runs, 2)
        self.assertEqual(
            [call.kwargs["custom_prices"].shape[1] for call in runner.call_args_list],
            [3, 4],
        )
        self.assertEqual([call.kwargs["k_target"] for call in runner.call_args_list], [3, 3])

    def test_live_market_sweep_fetches_largest_portfolio_once_then_slices(self):
        base = ExperimentConfig(
            data_mode="market",
            k_prior=1,
            k=1,
            k_target=3,
            all_tickers=["AAA", "BBB", "CCC", "DDD", "XXX"],
            selected_tickers=["AAA", "BBB", "CCC", "DDD"],
            target_date="2026-01-06",
            interval="1d",
        )
        prices = pd.DataFrame(
            {
                "AAA": [10.0, 11.0],
                "CCC": [30.0, 31.0],
                "DDD": [40.0, 41.0],
                "XXX": [50.0, 51.0],
            },
            index=pd.date_range("2026-01-01", periods=2, freq="D"),
        )
        loader = Mock(
            return_value=(
                prices,
                {
                    "requested_tickers": ["AAA", "BBB", "CCC", "DDD"],
                    "requested_count": 4,
                    "final_tickers": ["AAA", "CCC", "DDD", "XXX"],
                    "final_count": 4,
                    "removed": [
                        {"ticker": "BBB", "reason": "missing", "replacement_candidate": False}
                    ],
                    "replacements": ["XXX"],
                    "shortfall": 0,
                    "observations": 2,
                    "provider": "fixture",
                },
            )
        )
        captured = []

        def fake_prediction_runner(**kwargs):
            sliced_prices, report = kwargs["market_data_loader"](
                all_tickers=kwargs["all_tickers"],
                selected_tickers=kwargs["selected_tickers"],
                end_date=kwargs["target_date"],
                interval=kwargs["interval"],
                required_rows=kwargs["k_prior"] + kwargs["k"] + 1,
                replace_incomplete=kwargs["replace_incomplete"],
            )
            captured.append((list(sliced_prices.columns), report))
            return sample_prediction(k_target=kwargs["k_target"])

        with patch(
            "application.experiment_runner.run_prediction",
            side_effect=fake_prediction_runner,
        ):
            experiment = run_portfolio_size_experiment(
                base,
                [3, 4],
                market_data_loader=loader,
            )

        self.assertEqual(experiment.successful_runs, 2)
        loader.assert_called_once()
        self.assertEqual(
            loader.call_args.kwargs["selected_tickers"],
            ["AAA", "BBB", "CCC", "DDD"],
        )
        self.assertEqual(captured[0][0], ["AAA", "CCC", "XXX"])
        self.assertEqual(captured[1][0], ["AAA", "CCC", "DDD", "XXX"])
        self.assertEqual(captured[0][1]["requested_count"], 3)
        self.assertEqual(captured[0][1]["final_count"], 3)
        self.assertEqual(captured[0][1]["replacements"], ["XXX"])


class KWindowExperimentTests(unittest.TestCase):
    def test_builder_varies_k_and_holds_prior_window_fixed(self):
        base = sample_config()
        base.custom_prices = pd.DataFrame(
            {"AAA": range(10), "BBB": range(10), "CCC": range(10)},
            index=pd.date_range("2026-01-01", periods=10, freq="D"),
        )

        configurations = build_k_configurations(base, [1, 3, 5])

        self.assertEqual([config.k for config in configurations], [1, 3, 5])
        self.assertEqual([config.k_prior for config in configurations], [1, 1, 1])
        self.assertEqual([config.k_target for config in configurations], [3, 3, 3])
        self.assertEqual(
            [len(config.custom_prices) for config in configurations],
            [10, 10, 10],
        )

    def test_builder_rejects_invalid_or_too_large_k_values(self):
        base = sample_config()
        base.custom_prices = pd.DataFrame(
            {"AAA": range(5), "BBB": range(5), "CCC": range(5)},
            index=pd.date_range("2026-01-01", periods=5, freq="D"),
        )

        for values, message in [
            ([], "at least one K"),
            ([2, 2], "unique"),
            ([0], "at least 1"),
            ([1.5], "whole number"),
            ([4], "maximum K of 3"),
        ]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    build_k_configurations(base, values)

    def test_runner_passes_each_k_and_continues_after_a_failure(self):
        base = sample_config()
        base.custom_prices = pd.DataFrame(
            {"AAA": range(5), "BBB": range(5), "CCC": range(5)},
            index=pd.date_range("2026-01-01", periods=5, freq="D"),
        )
        runner = Mock(
            side_effect=[ValueError("K failed"), sample_prediction(k_target=3)]
        )

        experiment = run_k_experiment(base, [1, 2], prediction_runner=runner)

        self.assertEqual(experiment.total_runs, 2)
        self.assertEqual(experiment.failed_runs, 1)
        self.assertEqual(experiment.successful_runs, 1)
        self.assertEqual([run.config.k for run in experiment.runs], [1, 2])
        self.assertEqual(experiment.runs[0].error_message, "K failed")
        self.assertEqual([call.kwargs["k"] for call in runner.call_args_list], [1, 2])

    def test_live_market_k_sweep_fetches_enough_history_once(self):
        base = ExperimentConfig(
            data_mode="market",
            k_prior=2,
            k=1,
            k_target=3,
            all_tickers=["AAA", "BBB", "CCC"],
            selected_tickers=["AAA", "BBB", "CCC"],
            target_date="2026-01-06",
            interval="1d",
        )
        prices = pd.DataFrame(
            {ticker: range(10, 17) for ticker in ["AAA", "BBB", "CCC"]},
            index=pd.date_range("2026-01-01", periods=7, freq="D"),
        )
        loader = Mock(return_value=(prices, {"provider": "fixture"}))
        loaded_for_k = []

        def fake_prediction_runner(**kwargs):
            run_prices, _ = kwargs["market_data_loader"](
                all_tickers=kwargs["all_tickers"],
                selected_tickers=kwargs["selected_tickers"],
                end_date=kwargs["target_date"],
                interval=kwargs["interval"],
                required_rows=kwargs["k_prior"] + kwargs["k"] + 1,
                replace_incomplete=kwargs["replace_incomplete"],
            )
            loaded_for_k.append((kwargs["k"], run_prices))
            return sample_prediction(k_target=3)

        with patch(
            "application.experiment_runner.run_prediction",
            side_effect=fake_prediction_runner,
        ):
            experiment = run_k_experiment(
                base,
                [1, 4],
                market_data_loader=loader,
            )

        self.assertEqual(experiment.successful_runs, 2)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs["required_rows"], 7)
        self.assertEqual([k_value for k_value, _ in loaded_for_k], [1, 4])
        self.assertTrue(loaded_for_k[0][1].equals(loaded_for_k[1][1]))


class BExperimentTests(unittest.TestCase):
    def test_default_b_percent_range_is_minus_50_to_zero_by_two(self):
        values = build_b_percent_values()

        self.assertEqual(values[0], -50)
        self.assertEqual(values[-1], 0)
        self.assertEqual(len(values), 26)
        self.assertTrue(all(right - left == 2 for left, right in zip(values, values[1:])))

    def test_b_percent_range_validation_and_endpoint_inclusion(self):
        self.assertEqual(build_b_percent_values(-9, 0, 4), [-9, -5, -1, 0])

        for args, message in [
            ((-101, 0, 1), "between -100 and 0"),
            ((-10, 1, 1), "between -100 and 0"),
            ((0, -10, 1), "less than or equal"),
            ((-10, 0, 0), "at least 1"),
            ((-10.5, 0, 1), "whole numbers"),
        ]:
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, message):
                    build_b_percent_values(*args)

    def test_builder_varies_b_percent_and_holds_other_inputs_fixed(self):
        base = sample_config(k_target=3)

        configurations = build_b_configurations(base, [-50, -48, -46, 0])

        self.assertEqual(
            [config.b_percent for config in configurations],
            [-50.0, -48.0, -46.0, 0.0],
        )
        self.assertEqual([config.k for config in configurations], [base.k] * 4)
        self.assertEqual([config.k_prior for config in configurations], [base.k_prior] * 4)
        self.assertEqual([config.k_target for config in configurations], [base.k_target] * 4)

    def test_existing_experiment_builders_keep_b_fixed(self):
        base = sample_config(k_target=3)
        base.b_percent = -42
        base.custom_prices = pd.DataFrame(
            {"AAA": range(8), "BBB": range(8), "CCC": range(8), "DDD": range(8)},
            index=pd.date_range("2026-01-01", periods=8, freq="D"),
        )

        configurations = (
            build_k_target_configurations(base, [3, 4])
            + build_portfolio_size_configurations(base, [3, 4])
            + build_k_configurations(base, [1, 2])
        )

        self.assertTrue(all(config.b_percent == -42 for config in configurations))
        self.assertEqual(base.to_prediction_kwargs()["b_percent"], -42)

    def test_builder_rejects_out_of_range_nonfinite_and_duplicate_values(self):
        base = sample_config()

        for values, message in [
            ([], "at least one B"),
            ([-102], "between -100 and 0"),
            ([1], "between -100 and 0"),
            ([float("nan")], "between -100 and 0"),
            ([True], "between -100 and 0"),
            ([-10, -10.0], "unique"),
        ]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    build_b_configurations(base, values)

    def test_b_experiment_passes_each_percent_and_keeps_going_after_failure(self):
        base = sample_config()
        runner = Mock(
            side_effect=[ValueError("barrier failed"), sample_prediction(k_target=3)]
        )

        experiment = run_b_experiment(base, [-10, -8], prediction_runner=runner)

        self.assertEqual(experiment.total_runs, 2)
        self.assertEqual(experiment.failed_runs, 1)
        self.assertEqual(experiment.successful_runs, 1)
        self.assertEqual([run.config.b_percent for run in experiment.runs], [-10.0, -8.0])
        self.assertEqual(
            [call.kwargs["b_percent"] for call in runner.call_args_list],
            [-10.0, -8.0],
        )
        self.assertEqual(experiment.runs[0].error_message, "barrier failed")


if __name__ == "__main__":
    unittest.main()
