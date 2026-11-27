import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from application.pipeline import InsufficientObservationsError


class PredictorPageTests(unittest.TestCase):
    def make_results_page(self):
        page = AppTest.from_file("pages/1_Predictor_Engine.py").run(timeout=20)
        page.session_state["tickers"] = ["AAA", "BBB", "CCC"]
        page.session_state["step"] = 2
        page.run(timeout=20)
        self.assertFalse(page.exception)

        page.session_state["step"] = 3
        page.session_state["data_source"] = "Upload Custom CSV"
        page.session_state["custom_df"] = pd.DataFrame(
            {"AAA": range(1, 62), "BBB": range(2, 63), "CCC": range(3, 64)},
            index=pd.date_range("2026-01-01", periods=61, freq="D"),
        )
        page.session_state["K"] = 30
        page.session_state["K_prior"] = 30
        page.session_state["k_target"] = 3
        return page

    def test_insufficient_csv_history_is_shown_as_warning(self):
        page = self.make_results_page()

        with patch(
            "application.pipeline.run_prediction",
            side_effect=InsufficientObservationsError(
                "Insufficient CSV data: requested 61 observations, but only 60 complete observations are available."
            ),
        ) as run_prediction:
            page.run(timeout=20)

        self.assertFalse(page.exception)
        self.assertTrue(any("Insufficient CSV data" in item.value for item in page.warning))
        run_prediction.assert_called_once()

    def test_successful_csv_run_displays_results(self):
        page = self.make_results_page()
        model_output = {"q": 0.5, "p_star": 0.05, "I_q": 0.1, "crash_probability": 0.01}

        with patch("method.algorithm.run_algorithm", return_value=model_output) as run_algorithm:
            page.run(timeout=20)

        self.assertFalse(page.exception)
        self.assertIn("Simulation Results", [item.value for item in page.header])
        self.assertTrue(
            any("Model probability" in metric.label for metric in page.metric)
        )
        run_algorithm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
