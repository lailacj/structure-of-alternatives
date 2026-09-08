"""Tests for fold-wise Top-K and Top-p boundary selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from evaluate_set_variant_grid import out_of_fold_predictions, select_boundaries, summarize_log_scores  # noqa: E402
from build_set_variant_prediction_grid import _probability_records, _target_positions  # noqa: E402


class SetVariantCvTests(unittest.TestCase):
    def _grid(self) -> pd.DataFrame:
        rows = []
        # Candidate 1 is best in training fold 1; candidate 2 is best in fold 0.
        for variant in ("top_k", "top_p"):
            for boundary in (1.0, 2.0):
                for unit, fold, human_rate in [("a", 0, 1.0), ("b", 1, 0.0)]:
                    probability = 0.9 if boundary == 2.0 else 0.1
                    rows.append(
                        {
                            "variant": variant,
                            "boundary": boundary,
                            "analysis_dataset_id": "row_a",
                            "analysis_label": "Row A",
                            "analysis_unit_id": unit,
                            "cv_fold": fold,
                            "human_rate": human_rate,
                            "set_probability": probability,
                            "ordering_probability": 0.5,
                            "conjunction_probability": probability / 2,
                            "disjunction_probability": min(1.0, probability + 0.5),
                        }
                    )
        return pd.DataFrame(rows)

    def test_each_fold_selects_using_only_the_other_fold(self) -> None:
        selections = select_boundaries(self._grid())
        self.assertEqual(len(selections), 4)
        for variant in ("top_k", "top_p"):
            chosen = selections.loc[selections["variant"].eq(variant)].set_index("heldout_fold")
            self.assertEqual(float(chosen.loc[0, "selected_boundary"]), 1.0)
            self.assertEqual(float(chosen.loc[1, "selected_boundary"]), 2.0)

    def test_oof_predictions_use_one_selected_boundary_per_unit(self) -> None:
        grid = self._grid()
        out = out_of_fold_predictions(grid, select_boundaries(grid))
        self.assertEqual(len(out), 4)
        self.assertTrue(out.groupby(["variant", "analysis_unit_id"]).size().eq(1).all())

    def test_log_score_summary_balances_dataset_means(self) -> None:
        out = out_of_fold_predictions(self._grid(), select_boundaries(self._grid()))
        by_dataset, balanced = summarize_log_scores(out)
        self.assertEqual(len(by_dataset), 8)
        self.assertEqual(len(balanced), 8)
        self.assertTrue(balanced["dataset_count"].eq(1).all())
        self.assertTrue(np.isfinite(balanced["balanced_mean_oof_log_score"]).all())

    def test_top_p_uses_sampled_prefix_mass(self) -> None:
        records = _probability_records(
            np.array([[2, 0, 1, 3], [0, 1, 2, 3]]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            query_position=np.array([0, 2]),
            trigger_position=np.array([2, 1]),
            k_values=[1],
            p_values=[0.5],
        )
        top_p = next(record for record in records if record["variant"] == "top_p")
        self.assertAlmostEqual(top_p["set_probability"], 0.5)
        self.assertAlmostEqual(top_p["conjunction_probability"], 0.5)
        self.assertAlmostEqual(top_p["disjunction_probability"], 0.5)

    def test_target_positions_extend_one_shared_ordering(self) -> None:
        sampled = np.array([[0, 1], [1, 2]])
        positions = _target_positions(
            sampled,
            np.array([0.4, 0.3, 0.2, 0.1]),
            target_indices=[0, 1, 2, 3],
            rng=np.random.default_rng(11),
        )
        self.assertEqual(positions.shape, (2, 4))
        self.assertEqual(positions[0, 0], 0)
        self.assertEqual(positions[0, 1], 1)
        self.assertEqual(positions[1, 1], 0)
        self.assertEqual(positions[1, 2], 1)
        self.assertEqual(len(set(positions[0])), 4)
        self.assertEqual(len(set(positions[1])), 4)


if __name__ == "__main__":
    unittest.main()
