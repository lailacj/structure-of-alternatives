"""Tests for the canonical observation schema and novel-focus adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from build_focus_canonical_observations import (  # noqa: E402
    UNKNOWN_REVISION,
    build_focus_canonical_observations,
)
from canonical_observations import (  # noqa: E402
    summarize_canonical_observations,
    validate_canonical_observations,
)


def _human_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "story": ["story_a", "story_a", "story_a"],
            "cleaned_trigger": ["tea", "tea", "tea"],
            "cleaned_query": ["coffee", "coffee", "coffee"],
            "neg": [1, 0, 1],
        }
    )


def _score_data() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "story": "story_a",
                "trigger": "tea",
                "query": "coffee",
                "base_prefix": "They have ",
                "trigger_prefix": "They have tea but not",
                "base_logprob_sum": -4.0,
                "base_logprob_mean": -4.0,
                "base_query_token_count": 1,
                "base_tokenization_mode": "exact_concat",
                "but_not_logprob_sum": -2.0,
                "but_not_logprob_mean": -2.0,
                "but_not_query_token_count": 1,
                "but_not_tokenization_mode": "exact_concat",
            },
            {
                "story": "story_a",
                "trigger": "coffee",
                "query": "tea",
                "base_prefix": "They have ",
                "trigger_prefix": "They have coffee but not",
                "base_logprob_sum": -3.0,
                "base_logprob_mean": -3.0,
                "base_query_token_count": 1,
                "base_tokenization_mode": "exact_concat",
                "but_not_logprob_sum": -2.5,
                "but_not_logprob_mean": -2.5,
                "but_not_query_token_count": 1,
                "but_not_tokenization_mode": "exact_concat",
            },
        ]
    )


class CanonicalObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = build_focus_canonical_observations(
            _human_data(),
            _score_data(),
        )

    def test_focus_adapter_builds_valid_observation(self) -> None:
        validate_canonical_observations(self.observations)
        row = self.observations.iloc[0]

        self.assertEqual(len(self.observations), 1)
        self.assertEqual(row["human_yes"], 2)
        self.assertEqual(row["human_total"], 3)
        self.assertAlmostEqual(row["human_rate"], 2.0 / 3.0)
        self.assertEqual(row["query_logprob_sum"], -4.0)
        self.assertEqual(row["trigger_logprob_sum"], -3.0)
        self.assertEqual(row["model_revision"], UNKNOWN_REVISION)
        self.assertFalse(row["model_provenance_complete"])

    def test_summary_reports_incomplete_provenance(self) -> None:
        summary = summarize_canonical_observations(self.observations)

        self.assertEqual(summary.row_count, 1)
        self.assertEqual(summary.group_count, 1)
        self.assertEqual(summary.incomplete_provenance_rows, 1)
        self.assertEqual(summary.x_but_not_y_rows, 1)

    def test_duplicate_keys_are_rejected(self) -> None:
        duplicated = pd.concat([self.observations, self.observations], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate key"):
            validate_canonical_observations(duplicated)

    def test_inconsistent_human_rate_is_rejected(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "human_rate"] = 0.25

        with self.assertRaisesRegex(ValueError, "human_rate"):
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

    def test_complete_provenance_can_be_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete model provenance"):
            validate_canonical_observations(
                self.observations,
                require_complete_provenance=True,
            )

    def test_missing_trigger_score_is_rejected(self) -> None:
        scores = _score_data().iloc[[0]].copy()

        with self.assertRaisesRegex(ValueError, "trigger scores"):
            build_focus_canonical_observations(_human_data(), scores)

    def test_prompt_mismatch_is_rejected(self) -> None:
        scores = _score_data()
        scores.loc[scores["query"].eq("tea"), "base_prefix"] = "Different prompt "

        with self.assertRaisesRegex(ValueError, "same base prompt"):
            build_focus_canonical_observations(_human_data(), scores)

    def test_nonfinite_score_is_rejected(self) -> None:
        invalid = self.observations.copy()
        invalid.loc[0, "trigger_logprob_sum"] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_canonical_observations(invalid)


if __name__ == "__main__":
    unittest.main()
