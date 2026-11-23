import unittest
from unittest.mock import Mock

import pandas as pd

from application.pipeline import run_prediction


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
        self.assertEqual(run.data_report["provider"], "User-uploaded CSV")
        self.assertEqual(run.model_output["crash_probability"], 0.25)
        prior_inputs, active_inputs, asset_count, k_prior, k, k_target = runner.call_args.args
        self.assertEqual(asset_count, 3)
        self.assertEqual((k_prior, k, k_target), (2, 3, 3))
        self.assertEqual(len(prior_inputs[0]), 3)
        self.assertEqual(len(active_inputs[0]), 4)

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
