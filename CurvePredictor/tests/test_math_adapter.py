import unittest

import pandas as pd

from data_acess.math_adapter import df_to_math_inputs


def sample_prices(rows=6):
    return pd.DataFrame(
        {
            "AAA": [10, 11, 12, 13, 14, 15][:rows],
            "BBB": [20, 21, 22, 23, 24, 25][:rows],
            "CCC": [30, 31, 32, 33, 34, 35][:rows],
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="D"),
        dtype=float,
    )


class DataFrameToMathInputsTests(unittest.TestCase):
    def test_exact_minimum_rows_produce_expected_prior_and_active_windows(self):
        prices = sample_prices(rows=6)

        prior, active, asset_count = df_to_math_inputs(prices, k_prior=2, k=3)

        self.assertEqual(asset_count, 3)
        self.assertEqual(prior[0], [10.0, 11.0, 12.0])
        self.assertEqual(active[0], [12.0, 13.0, 14.0, 15.0])
        self.assertEqual(prior[0][-1], active[0][0])

    def test_one_fewer_than_required_rows_is_rejected(self):
        prices = sample_prices(rows=5)

        with self.assertRaisesRegex(ValueError, r"6 days required.*DataFrame has 5"):
            df_to_math_inputs(prices, k_prior=2, k=3)


if __name__ == "__main__":
    unittest.main()
