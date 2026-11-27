import unittest
import sys
import types
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from data_acess.provider import (
    InsufficientHistoricalDataError,
    NoMarketDataError,
    _period_days,
    assess_price_matrix,
    fetch_stock_data,
    get_sliced_portfolio,
    validate_dataframe,
)


class ValidateDataFrameTests(unittest.TestCase):
    def test_valid_wide_csv_is_normalized(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "AAA": [10, 11], "BBB": [20, 21]})
        result = validate_dataframe(frame)
        self.assertEqual(list(result.columns), ["AAA", "BBB"])
        self.assertEqual(result.index.name, "Date")
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_missing_prices_are_rejected_for_uploaded_csv(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "AAA": [10, None]})
        with self.assertRaisesRegex(ValueError, "missing or non-numeric"):
            validate_dataframe(frame)

    def test_non_positive_prices_are_rejected(self):
        frame = pd.DataFrame({"date": ["2026-01-01"], "AAA": [0]})
        with self.assertRaisesRegex(ValueError, "zero or negative"):
            validate_dataframe(frame)


class AssessPriceMatrixTests(unittest.TestCase):
    def test_excludes_missing_or_short_assets_without_imputation(self):
        prices = pd.DataFrame({
            "GOOD": [10.0, 11.0, 12.0],
            "GAP": [10.0, np.nan, 12.0],
            "SHORT": [10.0, 11.0, np.nan],
        })
        eligible, reasons = assess_price_matrix(prices, required_rows=3)
        self.assertEqual(list(eligible.columns), ["GOOD"])
        self.assertIn("GAP", reasons)
        self.assertIn("missing observations", reasons["GAP"])
        self.assertIn("SHORT", reasons)

    def test_no_history_has_explicit_reason(self):
        prices = pd.DataFrame({"NONE": [np.nan, np.nan]})
        eligible, reasons = assess_price_matrix(prices, required_rows=2)
        self.assertTrue(eligible.empty)
        self.assertEqual(reasons["NONE"], "no price history returned")


class HistoricalWindowTests(unittest.TestCase):
    def test_large_hourly_window_is_bounded_below_provider_limit(self):
        self.assertEqual(_period_days("1h", 10001), 728)

    def test_nonempty_short_history_reports_requested_and_available_rows(self):
        prices = pd.DataFrame(
            {ticker: np.arange(12, dtype=float) + 10 for ticker in ("AAA", "BBB", "CCC")},
            index=pd.date_range("2026-01-01", periods=12, freq="h"),
        )
        with patch("data_acess.provider.fetch_stock_data", return_value=prices) as fetch:
            with self.assertRaisesRegex(
                InsufficientHistoricalDataError,
                r"requested 100 observations, but only 12 complete observations.*1h",
            ):
                get_sliced_portfolio(
                    all_tickers=["AAA", "BBB", "CCC"],
                    selected_tickers=["AAA", "BBB", "CCC"],
                    end_date="2026-01-01",
                    interval="1h",
                    required_rows=100,
                )
        fetch.assert_called_once()

    def test_empty_provider_result_has_interval_and_date_range(self):
        download = Mock(return_value=pd.DataFrame())
        fake_yfinance = types.SimpleNamespace(download=download)
        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            with self.assertRaisesRegex(
                NoMarketDataError,
                r"no data for interval '1h'.*from .* through .*provider may not offer data",
            ):
                fetch_stock_data(["AAA"], "2026-11-23", interval="1h", required_rows=10001)

        self.assertEqual(download.call_args.kwargs["interval"], "1h")
        self.assertEqual(download.call_args.kwargs["start"], "2024-11-25")
        self.assertEqual(download.call_args.kwargs["end"], "2026-11-24")


if __name__ == "__main__":
    unittest.main()
