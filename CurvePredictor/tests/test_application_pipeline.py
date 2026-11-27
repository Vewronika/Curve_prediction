import unittest
from unittest.mock import Mock

import pandas as pd

from application.pipeline import InsufficientObservationsError, run_prediction
from data_acess.provider import InsufficientHistoricalDataError, NoMarketDataError


def sample_prices(rows=6):
    return pd.DataFrame(
        {
            "AAA": range(10, 10 + rows),
            "BBB": range(20, 20 + rows),
            "CCC": range(30, 30 + rows),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="D"),
        dtype=float,
    )


class RunPredictionTests(unittest.TestCase):
    def test_csv_data_is_adapted_and_sent_to_model_runner(self):
        prices = sample_prices()
        runner = Mock(return_value={"crash_probability": 0.25})

        run = run_prediction(
            data_mode="csv",
            custom_prices=prices,
            k_prior=2,
            k=3,
            k_target=3,
            model_runner=runner,
        )

        self.assertEqual(run.asset_count, 3)
        self.assertEqual(run.b_percent, -10.0)
        self.assertTrue(run.prices.equals(prices))
        self.assertEqual(run.data_report["provider"], "User-uploaded CSV")
        self.assertEqual(run.model_output["crash_probability"], 0.25)
        prior_inputs, active_inputs, asset_count, k_prior, k, k_target = runner.call_args.args
        self.assertEqual(asset_count, 3)
        self.assertEqual((k_prior, k, k_target), (2, 3, 3))
        self.assertEqual(runner.call_args.kwargs["B"], -0.1)
        self.assertEqual(prior_inputs[0], [10.0, 11.0, 12.0])
        self.assertEqual(active_inputs[0], [12.0, 13.0, 14.0, 15.0])

    def test_b_percent_is_converted_to_algorithm_scale(self):
        for b_percent, expected_b in [(-100, -1.0), (-50, -0.5), (0, 0.0)]:
            with self.subTest(b_percent=b_percent):
                runner = Mock(return_value={"crash_probability": 0.25})

                run = run_prediction(
                    data_mode="csv",
                    custom_prices=sample_prices(),
                    k_prior=2,
                    k=3,
                    k_target=3,
                    b_percent=b_percent,
                    model_runner=runner,
                )

                self.assertEqual(runner.call_args.kwargs["B"], expected_b)
                self.assertEqual(run.b_percent, float(b_percent))

    def test_invalid_b_percent_is_rejected_before_loading_market_data(self):
        loader = Mock()
        runner = Mock()

        for b_percent in [-100.1, 0.1, float("nan"), float("inf"), True, "-10"]:
            with self.subTest(b_percent=b_percent):
                with self.assertRaisesRegex(ValueError, "B must be"):
                    run_prediction(
                        data_mode="market",
                        all_tickers=["AAA", "BBB", "CCC"],
                        selected_tickers=["AAA", "BBB", "CCC"],
                        target_date="2026-01-06",
                        interval="1d",
                        k_prior=2,
                        k=3,
                        k_target=3,
                        b_percent=b_percent,
                        market_data_loader=loader,
                        model_runner=runner,
                    )

        loader.assert_not_called()
        runner.assert_not_called()

    def test_market_data_loader_receives_required_window(self):
        prices = sample_prices()
        loader = Mock(return_value=(prices, {"provider": "fixture"}))
        runner = Mock(return_value={"crash_probability": 0.1})

        run_prediction(
            data_mode="market",
            all_tickers=["AAA", "BBB", "CCC"],
            selected_tickers=["AAA", "BBB", "CCC"],
            target_date="2026-01-06",
            interval="1d",
            k_prior=2,
            k=3,
            k_target=3,
            market_data_loader=loader,
            model_runner=runner,
        )

        self.assertEqual(loader.call_args.kwargs["required_rows"], 6)
        self.assertEqual(loader.call_args.kwargs["interval"], "1d")
        runner.assert_called_once()

    def test_short_csv_reports_actual_complete_observations_before_model(self):
        runner = Mock()

        with self.assertRaisesRegex(
            InsufficientObservationsError,
            r"requested 6 observations, but only 5 complete observations",
        ):
            run_prediction(
                data_mode="csv",
                custom_prices=sample_prices(rows=5),
                k_prior=2,
                k=3,
                k_target=3,
                model_runner=runner,
            )

        runner.assert_not_called()

    def test_short_live_history_has_specific_provider_exception(self):
        loader = Mock(return_value=(sample_prices(rows=5), {"provider": "fixture"}))
        runner = Mock()

        with self.assertRaisesRegex(
            InsufficientHistoricalDataError,
            r"requested 6 observations, but only 5 complete observations.*1d",
        ):
            run_prediction(
                data_mode="market",
                all_tickers=["AAA", "BBB", "CCC"],
                selected_tickers=["AAA", "BBB", "CCC"],
                target_date="2026-01-05",
                interval="1d",
                k_prior=2,
                k=3,
                k_target=3,
                market_data_loader=loader,
                model_runner=runner,
            )

        runner.assert_not_called()

    def test_empty_market_result_exception_is_preserved(self):
        loader = Mock(side_effect=NoMarketDataError("no rows for this range"))

        with self.assertRaisesRegex(NoMarketDataError, "no rows for this range"):
            run_prediction(
                data_mode="market",
                all_tickers=["AAA", "BBB", "CCC"],
                selected_tickers=["AAA", "BBB", "CCC"],
                target_date="2026-01-06",
                interval="1d",
                k_prior=2,
                k=3,
                k_target=3,
                market_data_loader=loader,
            )

    def test_fractional_window_is_not_silently_truncated(self):
        loader = Mock()

        with self.assertRaisesRegex(ValueError, "K must be a whole number"):
            run_prediction(
                data_mode="market",
                all_tickers=["AAA", "BBB", "CCC"],
                selected_tickers=["AAA", "BBB", "CCC"],
                target_date="2026-01-06",
                interval="1d",
                k_prior=2,
                k=3.5,
                k_target=3,
                market_data_loader=loader,
            )

        loader.assert_not_called()

    def test_invalid_threshold_is_rejected_before_model(self):
        runner = Mock()

        with self.assertRaisesRegex(ValueError, "Crash threshold k=3.5 is invalid"):
            run_prediction(
                data_mode="csv",
                custom_prices=sample_prices(),
                k_prior=2,
                k=3,
                k_target=3.5,
                model_runner=runner,
            )

        runner.assert_not_called()

    def test_invalid_portfolio_stops_before_model_runner(self):
        runner = Mock()
        prices = sample_prices().iloc[:, :2]

        with self.assertRaisesRegex(ValueError, "At least 3 are required"):
            run_prediction(
                data_mode="csv",
                custom_prices=prices,
                k_prior=2,
                k=3,
                k_target=3,
                model_runner=runner,
            )

        runner.assert_not_called()

    def test_invalid_window_stops_before_market_data_load(self):
        loader = Mock()

        with self.assertRaisesRegex(ValueError, "Both time windows must be at least 1"):
            run_prediction(
                data_mode="market",
                all_tickers=["AAA", "BBB", "CCC"],
                selected_tickers=["AAA", "BBB", "CCC"],
                target_date="2026-01-06",
                interval="1d",
                k_prior=0,
                k=3,
                k_target=3,
                market_data_loader=loader,
            )

        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
