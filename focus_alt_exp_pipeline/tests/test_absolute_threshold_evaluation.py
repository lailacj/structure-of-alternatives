"""Tests for grouped absolute-threshold fitting and evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from absolute_threshold_models import structure_probabilities  # noqa: E402
from canonical_observations import (  # noqa: E402
    CANONICAL_COLUMNS,
    SCHEMA_VERSION,
    validate_canonical_observations,
)
from evaluate_absolute_threshold import (  # noqa: E402
    STRUCTURES,
    fit_shared_threshold,
    leave_one_group_out_predictions,
    summarize_out_of_fold_predictions,
)


def _synthetic_observations() -> tuple[pd.DataFrame, float]:
    true_threshold = -3.0
    query_scores = np.linspace(-8.0, -0.5, 16)
    trigger_scores = np.linspace(-6.0, -1.0, 16)
    set_probabilities = np.asarray(
        structure_probabilities(
            query_scores,
            trigger_scores,
            threshold=true_threshold,
        ).set
    )
    human_total = np.full(len(query_scores), 10_000, dtype=int)
    human_yes = np.rint(set_probabilities * human_total).astype(int)

    records = []
    for index, (query_score, trigger_score, yes, total) in enumerate(
        zip(query_scores, trigger_scores, human_yes, human_total)
    ):
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_family": "synthetic",
                "dataset": "threshold_recovery",
                "condition": "test",
                "item_id": f"item_{index:02d}",
                "group_id": f"group_{index // 2:02d}",
                "context_id": f"context_{index // 2:02d}",
                "generation_frame": "no_frame",
                "generation_prompt": f"Context {index // 2}: ",
                "trigger": f"trigger_{index}",
                "query": f"query_{index}",
                "human_outcome": "query_excluded",
                "human_yes": int(yes),
                "human_total": int(total),
                "human_rate": float(yes / total),
                "human_count_status": "exact",
                "trigger_logprob_sum": float(trigger_score),
                "query_logprob_sum": float(query_score),
                "trigger_logprob_mean": float(trigger_score),
                "query_logprob_mean": float(query_score),
                "trigger_token_count": 1,
                "query_token_count": 1,
                "trigger_tokenization_mode": "synthetic",
                "query_tokenization_mode": "synthetic",
                "x_but_not_y_applicable": False,
                "x_but_not_y_prompt": np.nan,
                "x_but_not_y_logprob_sum": np.nan,
                "x_but_not_y_logprob_mean": np.nan,
                "x_but_not_y_token_count": np.nan,
                "x_but_not_y_tokenization_mode": np.nan,
                "model_name": "synthetic",
                "model_revision": "test",
                "model_provenance_complete": True,
                "source_human_file": "synthetic_human.csv",
                "source_score_file": "synthetic_scores.csv",
            }
        )
    observations = pd.DataFrame.from_records(records, columns=CANONICAL_COLUMNS)
    validate_canonical_observations(observations)
    return observations, true_threshold


class AbsoluteThresholdEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations, self.true_threshold = _synthetic_observations()

    def test_set_fit_recovers_known_threshold(self) -> None:
        fitted = fit_shared_threshold(self.observations, fit_target="set")

        self.assertAlmostEqual(fitted.threshold, self.true_threshold, delta=0.01)
        self.assertFalse(fitted.at_boundary)

    def test_leave_one_group_out_predicts_each_item_once(self) -> None:
        predictions, folds = leave_one_group_out_predictions(self.observations)

        self.assertEqual(len(predictions), len(self.observations))
        self.assertEqual(len(folds), self.observations["group_id"].nunique())
        self.assertFalse(
            predictions.duplicated(["dataset_family", "dataset", "condition", "item_id"]).any()
        )
        self.assertTrue(predictions["fold_id"].notna().all())

    def test_fold_threshold_excludes_heldout_group(self) -> None:
        predictions, folds = leave_one_group_out_predictions(self.observations)
        first_fold = folds.iloc[0]
        heldout_group_key = first_fold["heldout_group_key"]
        group_keys = (
            self.observations["dataset_family"]
            + "::"
            + self.observations["dataset"]
            + "::"
            + self.observations["group_id"]
        )
        manual_fit = fit_shared_threshold(
            self.observations.loc[~group_keys.eq(heldout_group_key)].copy()
        )

        self.assertAlmostEqual(first_fold["threshold"], manual_fit.threshold)
        heldout_predictions = predictions.loc[
            predictions["heldout_group_key"].eq(heldout_group_key)
        ]
        self.assertFalse(heldout_predictions.empty)

    def test_output_probabilities_use_the_fold_threshold(self) -> None:
        predictions, _ = leave_one_group_out_predictions(self.observations)
        row = predictions.iloc[0]
        expected = structure_probabilities(
            row["query_logprob_sum"],
            row["trigger_logprob_sum"],
            threshold=row["threshold"],
            scale=row["gumbel_scale"],
        )

        for structure in STRUCTURES:
            self.assertAlmostEqual(
                row[f"{structure}_probability"],
                getattr(expected, structure),
            )

    def test_summary_contains_correlations_and_log_scores(self) -> None:
        predictions, _ = leave_one_group_out_predictions(self.observations)
        summary = summarize_out_of_fold_predictions(predictions)

        overall = summary.loc[summary["scope_type"].eq("overall")]
        self.assertEqual(set(overall["structure"]), set(STRUCTURES))
        self.assertTrue(np.isfinite(overall["mean_log_score"]).all())
        self.assertTrue(np.isfinite(overall["baseline_mean_log_score"]).all())
        self.assertTrue(np.isfinite(overall["pearson_r"]).all())
        self.assertTrue(np.isfinite(overall["spearman_rho"]).all())


if __name__ == "__main__":
    unittest.main()
