"""Tests for the canonical observation schema and standardized focus adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from build_focus_canonical_observations import (  # noqa: E402
    build_focus_canonical_observations,
)
from canonical_observations import (  # noqa: E402
    summarize_canonical_observations,
    validate_canonical_observations,
)


REVISION = "453ed1575b739b5b03ce3758b23befdb0967f40e"


def _human_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "story": ["story_a", "story_a", "story_a"],
            "cleaned_trigger": ["tea", "tea", "tea"],
            "cleaned_query": ["coffee", "coffee", "coffee"],
            "neg": [1, 0, 1],
        }
    )


def _score_row(*, frame: str) -> dict[str, object]:
    no_frame = frame == "no_frame"
    return {
        "dataset_family": "novel_focus",
        "dataset": "focus_alternative_study",
        "condition": "focus_only",
        "item_id": f"story_a::tea::coffee::{frame}",
        "group_id": "story_a",
        "context_id": "story_a",
        "generation_frame": frame,
        "generation_prompt": "They have " if no_frame else "They have tea but not ",
        "trigger": "tea",
        "query": "coffee",
        "analysis_inclusion_status": "included",
        "source_row_id": "story_a::tea::coffee",
        "trigger_logprob_sum": -3.0 if no_frame else np.nan,
        "trigger_logprob_mean": -3.0 if no_frame else np.nan,
        "trigger_token_count": 1 if no_frame else np.nan,
        "trigger_tokenization_mode": "exact_concat" if no_frame else np.nan,
        "query_logprob_sum": -4.0 if no_frame else -2.0,
        "query_logprob_mean": -4.0 if no_frame else -2.0,
        "query_token_count": 1,
        "query_tokenization_mode": "exact_concat",
        "model_identifier": "Qwen/Qwen2-7B",
        "model_revision": REVISION,
        "model_revision_source": "test",
        "resolved_model_path": f"/cache/snapshots/{REVISION}",
        "actual_model_dtype": "bfloat16",
        "torch_version": "test",
        "transformers_version": "test",
    }


def _score_data() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [_score_row(frame="no_frame"), _score_row(frame="x_but_not_y")]
    )


class CanonicalObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = build_focus_canonical_observations(
            _human_data(),
            _score_data(),
            expected_rows=1,
        )

    def test_focus_adapter_builds_valid_observation(self) -> None:
        validate_canonical_observations(self.observations)
        row = self.observations.iloc[0]

        self.assertEqual(len(self.observations), 1)
        self.assertEqual(row["human_yes"], 2)
        self.assertEqual(row["human_total"], 3)
        self.assertAlmostEqual(row["human_rate"], 2.0 / 3.0)
        self.assertEqual(row["human_count_status"], "exact")
        self.assertEqual(row["query_logprob_sum"], -4.0)
        self.assertEqual(row["trigger_logprob_sum"], -3.0)
        self.assertEqual(row["x_but_not_y_logprob_sum"], -2.0)
        self.assertEqual(row["model_revision"], REVISION)
        self.assertTrue(row["model_provenance_complete"])

    def test_summary_reports_count_availability(self) -> None:
        summary = summarize_canonical_observations(self.observations)

        self.assertEqual(summary.row_count, 1)
        self.assertEqual(summary.group_count, 1)
        self.assertEqual(summary.incomplete_provenance_rows, 0)
        self.assertEqual(summary.x_but_not_y_rows, 1)
        self.assertEqual(summary.exact_count_rows, 1)
        self.assertEqual(summary.rate_only_rows, 0)

    def test_duplicate_keys_are_rejected(self) -> None:
        duplicated = pd.concat([self.observations, self.observations], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate key"):
            validate_canonical_observations(duplicated)

    def test_inconsistent_human_rate_is_rejected(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "human_rate"] = 0.25

        with self.assertRaisesRegex(ValueError, "human_rate"):
            validate_canonical_observations(invalid)

    def test_rate_only_rows_require_missing_counts(self) -> None:
        rate_only = self.observations.copy()
        rate_only.loc[0, ["human_yes", "human_total"]] = np.nan
        rate_only.loc[0, "human_count_status"] = "rate_only"

        validate_canonical_observations(rate_only)
        with self.assertRaisesRegex(ValueError, "rate-only rows"):
            validate_canonical_observations(rate_only, require_human_counts=True)

    def test_rate_only_rows_reject_fabricated_counts(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "human_count_status"] = "rate_only"

        with self.assertRaisesRegex(ValueError, "fabricated"):
            validate_canonical_observations(invalid)

    def test_inconsistent_mean_logprob_is_rejected(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "query_logprob_mean"] = -1.0

        with self.assertRaisesRegex(ValueError, "query mean"):
            validate_canonical_observations(invalid)

    def test_nonapplicable_x_but_not_y_rows_cannot_keep_scores(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "x_but_not_y_applicable"] = False

        with self.assertRaisesRegex(ValueError, "must not contain scores"):
            validate_canonical_observations(invalid)

    def test_missing_trigger_score_is_rejected(self) -> None:
        scores = _score_data()
        scores.loc[scores["generation_frame"].eq("no_frame"), "trigger_logprob_sum"] = np.nan

        with self.assertRaisesRegex(ValueError, "missing trigger scores"):
            build_focus_canonical_observations(
                _human_data(),
                scores,
                expected_rows=1,
            )

    def test_mixed_model_revisions_are_rejected(self) -> None:
        scores = _score_data()
        scores.loc[scores["generation_frame"].eq("x_but_not_y"), "model_revision"] = "other"

        with self.assertRaisesRegex(ValueError, "one non-empty model_revision"):
            build_focus_canonical_observations(
                _human_data(),
                scores,
                expected_rows=1,
            )

    def test_nonfinite_score_is_rejected(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "trigger_logprob_sum"] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_canonical_observations(invalid)


if __name__ == "__main__":
    unittest.main()
