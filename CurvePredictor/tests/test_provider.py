import unittest
import io
import sys
import types
from unittest.mock import Mock, call, patch
import numpy as np
import pandas as pd

from data_acess.provider import (
    InsufficientHistoricalDataError,
    NoMarketDataError,
    _period_days,
    assess_price_matrix,
    fetch_stock_data,
    get_sliced_portfolio,
    parse_uploaded_csv,
    validate_dataframe,
)


class ValidateDataFrameTests(unittest.TestCase):
    def test_header_only_uploaded_csv_is_rejected(self):
        csv_file = io.StringIO("date,AAA,BBB\n")

        with self.assertRaisesRegex(ValueError, "uploaded CSV is empty"):
            parse_uploaded_csv(csv_file)

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

    def test_zero_byte_csv_becomes_a_clear_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read the CSV file"):
            parse_uploaded_csv(io.StringIO(""))

    def test_duplicate_dates_are_rejected(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-01"], "AAA": [10, 11]})
        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            validate_dataframe(frame)

    def test_dates_must_be_chronological(self):
        frame = pd.DataFrame({"date": ["2026-01-02", "2026-01-01"], "AAA": [10, 11]})
        with self.assertRaisesRegex(ValueError, "chronological order"):
            validate_dataframe(frame)

    def test_duplicate_headers_after_trimming_are_rejected(self):
        frame = pd.DataFrame([["2026-01-01", 10, 20]], columns=["date", "AAA", " AAA "])
        with self.assertRaisesRegex(ValueError, "must be unique after trimming"):
            validate_dataframe(frame)

    def test_infinite_prices_are_rejected(self):
        frame = pd.DataFrame({"date": ["2026-01-01"], "AAA": [np.inf]})
        with self.assertRaisesRegex(ValueError, "missing or non-numeric"):
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


class PortfolioAssemblyTests(unittest.TestCase):
    def test_replaces_unavailable_asset_and_reports_final_portfolio(self):
        index = pd.date_range("2026-01-01", periods=3, freq="D")
        initial = pd.DataFrame(
            {"AAA": [10.0, 11.0, 12.0], "BBB": [20.0, 21.0, 22.0], "BAD": [np.nan] * 3},
            index=index,
        )
        candidate = pd.DataFrame({"DDD": [30.0, 31.0, 32.0]}, index=index)
        final = pd.DataFrame(
            {"AAA": [10.0, 11.0, 12.0], "BBB": [20.0, 21.0, 22.0], "DDD": [30.0, 31.0, 32.0]},
            index=index,
        )

        with patch(
            "data_acess.provider.fetch_stock_data",
            side_effect=[initial, candidate, final],
        ) as fetch:
            prices, report = get_sliced_portfolio(
                all_tickers=["AAA", "BBB", "BAD", "DDD"],
                selected_tickers=["AAA", "BBB", "BAD"],
                end_date="2026-01-03",
                interval="1d",
                required_rows=3,
            )

        self.assertEqual(list(prices.columns), ["AAA", "BBB", "DDD"])
        self.assertEqual(report["replacements"], ["DDD"])
        self.assertEqual(report["observations"], 3)
        fetch.assert_has_calls([
            call(["AAA", "BBB", "BAD"], "2026-01-03", "1d", 3),
            call(["DDD"], "2026-01-03", "1d", 3),
            call(["AAA", "BBB", "DDD"], "2026-01-03", "1d", 3),
        ])

    def test_final_alignment_reports_insufficient_complete_rows(self):
        index = pd.date_range("2026-01-01", periods=4, freq="D")
        initial = pd.DataFrame(
            {"AAA": [10.0, 11.0, 12.0, 13.0], "BBB": [20.0, 21.0, 22.0, 23.0], "CCC": [30.0, 31.0, 32.0, 33.0]},
            index=index,
        )
        final = initial.copy()
        final.loc[index[1], "BBB"] = np.nan

        with patch("data_acess.provider.fetch_stock_data", side_effect=[initial, final]):
            with self.assertRaisesRegex(
                InsufficientHistoricalDataError,
                r"requested 4 observations, but only 3 complete observations",
            ):
                get_sliced_portfolio(
                    all_tickers=["AAA", "BBB", "CCC"],
                    selected_tickers=["AAA", "BBB", "CCC"],
                    end_date="2026-01-04",
                    interval="1d",
                    required_rows=4,
                    replace_incomplete=False,
                )


if __name__ == "__main__":
    unittest.main()
