import contextlib
import io
import math
import unittest

import pandas as pd

from application.pipeline import run_prediction


class SingleRunIntegrationTests(unittest.TestCase):
    def test_csv_flows_through_adapter_and_real_algorithm(self):
        # Small, deterministic fixture: 5 observations support K_prior=2 and K=2.
        prices = pd.DataFrame(
            {
                "AAA": [100, 110, 108, 104, 95],
                "BBB": [100, 95, 92, 100, 105],
                "CCC": [100, 102, 103, 110, 108],
                "DDD": [100, 106, 99, 96, 101],
                "EEE": [100, 98, 105, 109, 103],
                "FFF": [100, 104, 97, 102, 111],
            },
            index=pd.date_range("2026-01-01", periods=5, freq="D"),
            dtype=float,
        )

        # The algorithm prints progress information; keep test output readable.
        with contextlib.redirect_stdout(io.StringIO()):
            run = run_prediction(
                data_mode="csv",
                custom_prices=prices,
                k_prior=2,
                k=2,
                k_target=3,
            )

        expected_keys = {
            "k", "N", "q", "p_star", "I_q", "exponent",
            "underflow_flag", "crash_probability", "log_crash_probability",
        }
        self.assertTrue(expected_keys.issubset(run.model_output))
        self.assertEqual(run.asset_count, 6)
        self.assertEqual(run.model_output["N"], 6)
        self.assertEqual(run.model_output["k"], 3)
        self.assertEqual(run.model_output["q"], 0.5)
        self.assertTrue(math.isfinite(run.model_output["p_star"]))
        self.assertTrue(math.isfinite(run.model_output["I_q"]))
        self.assertTrue(0.0 <= run.model_output["crash_probability"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
