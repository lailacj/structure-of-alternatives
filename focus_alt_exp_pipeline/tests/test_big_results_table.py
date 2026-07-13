"""Regression and leakage checks for the cross-dataset development table."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PIPELINE_DIR / "code"
RESULT_DIR = PIPELINE_DIR / "results" / "big_table_development"
sys.path.insert(0, str(CODE_DIR))

from build_big_results_table import (  # noqa: E402
    ANALYSIS_ROWS,
    EXPECTED_ANALYSIS_UNITS,
    HU_ORIGINAL_EXPECTEDNESS_R,
    STRUCTURES,
)


class BigResultsTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.units = pd.read_csv(RESULT_DIR / "analysis_units.csv")
        cls.predictions = pd.read_csv(RESULT_DIR / "oof_predictions.csv")
        cls.folds = pd.read_csv(RESULT_DIR / "fold_parameters.csv")
        cls.correlations = pd.read_csv(RESULT_DIR / "big_table_correlations.csv")
        cls.log_scores = pd.read_csv(RESULT_DIR / "big_table_log_scores.csv")
        cls.metrics = pd.read_csv(RESULT_DIR / "big_table_metrics.csv")

    def test_analysis_grain_and_table_order_are_frozen(self) -> None:
        self.assertEqual(len(self.units), sum(EXPECTED_ANALYSIS_UNITS.values()))
        self.assertEqual(len(self.units), 993)
        self.assertEqual(
            self.units.groupby("analysis_dataset_id").size().to_dict(),
            EXPECTED_ANALYSIS_UNITS,
        )
        expected_order = [row_id for row_id, _ in ANALYSIS_ROWS]
        self.assertEqual(self.correlations["analysis_dataset_id"].tolist(), expected_order)
        self.assertEqual(self.metrics["analysis_dataset_id"].tolist(), expected_order)

    def test_grouped_folds_prevent_source_item_leakage(self) -> None:
        self.assertEqual(self.units.groupby("cv_group_id")["cv_fold"].nunique().max(), 1)
        rnx = self.units.loc[self.units["dataset_family"].eq("ronai_xiang_2024")]
        self.assertTrue(rnx.groupby("cv_group_id")["analysis_dataset_id"].nunique().eq(5).all())
        focus = self.units.loc[self.units["dataset_family"].eq("novel_focus")]
        self.assertEqual(focus["cv_group_id"].nunique(), 16)

    def test_every_unit_has_one_valid_heldout_prediction(self) -> None:
        keys = ["analysis_dataset_id", "analysis_unit_id"]
        self.assertEqual(len(self.predictions), len(self.units))
        self.assertFalse(self.predictions.duplicated(keys).any())
        probability_columns = [
            "intercept_probability",
            *[f"{structure}_probability" for structure in STRUCTURES],
        ]
        probabilities = self.predictions[probability_columns].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        self.assertTrue(
            (
                self.predictions["conjunction_probability"]
                <= self.predictions[["set_probability", "ordering_probability"]].min(axis=1)
                + 1e-12
            ).all()
        )
        self.assertTrue(
            (
                self.predictions["disjunction_probability"] + 1e-12
                >= self.predictions[["set_probability", "ordering_probability"]].max(axis=1)
            ).all()
        )

    def test_hu_original_and_qwen_raw_correlations_lock_the_source_grain(self) -> None:
        hu = self.correlations.set_index("analysis_dataset_id").loc[
            ["hu_rx22", "hu_pvt21", "hu_g18", "hu_vt16"]
        ]
        self.assertEqual(hu["N"].astype(int).tolist(), [57, 50, 67, 39])
        for row_id, expected in HU_ORIGINAL_EXPECTEDNESS_R.items():
            self.assertAlmostEqual(
                float(hu.loc[row_id, "hu_original_expectedness_r"]),
                expected,
                places=12,
            )
        self.assertAlmostEqual(
            float(hu.loc["hu_vt16", "qwen_x_but_not_y_r"]),
            0.08669912155636568,
            places=12,
        )
        self.assertAlmostEqual(
            float(hu.loc["hu_vt16", "qwen_no_frame_r"]),
            0.12805917005925782,
            places=12,
        )
        rnx = self.correlations["analysis_dataset_id"].str.startswith("rnx_")
        self.assertTrue(self.correlations.loc[rnx, "qwen_x_but_not_y_r"].isna().all())

    def test_fit_diagnostics_are_successful_and_interior(self) -> None:
        self.assertEqual(len(self.folds), 10)
        self.assertTrue(self.folds["optimizer_success"].all())
        self.assertFalse(self.folds["threshold_at_boundary"].any())
        self.assertFalse(self.folds["scale_at_boundary"].any())
        self.assertTrue(self.folds["gumbel_scale"].between(0.1, 20.0).all())

    def test_log_score_availability_respects_source_counts(self) -> None:
        self.assertEqual(len(self.log_scores), len(ANALYSIS_ROWS) * 5)
        rate_only = {"hu_pvt21", "hu_g18", "hu_vt16"}
        for row_id, rows in self.log_scores.groupby("analysis_dataset_id"):
            if row_id in rate_only:
                self.assertTrue(rows["response_total_log_likelihood"].isna().all())
                self.assertTrue(rows["response_count"].isna().all())
            else:
                self.assertTrue(rows["response_total_log_likelihood"].notna().all())
                self.assertTrue(rows["response_count"].notna().all())
        intercept = self.log_scores.loc[self.log_scores["model"].eq("intercept")]
        self.assertTrue(
            np.allclose(intercept["delta_item_log_score_vs_intercept"], 0.0, atol=1e-15)
        )


if __name__ == "__main__":
    unittest.main()
