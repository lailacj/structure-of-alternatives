"""Tests for the Hu/R&X no-frame scoring manifest."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PIPELINE_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

from scoring_manifest import (  # noqa: E402
    FRAME_AWARE_MANIFEST_VERSION,
    MANIFEST_COLUMNS,
    SCORING_CONTROL_COLUMNS,
    summarize_scoring_manifest,
    validate_scoring_manifest,
)


def _row(
    *,
    family: str,
    condition: str,
    item_id: str,
    group_id: str,
    prompt: str,
    trigger: str = "warm",
    query: str = "hot",
) -> dict[str, object]:
    datasets = {
        "hu_2023_benchmark": "toy_hu",
        "ronai_xiang_2024": "ronai_xiang_2024",
        "novel_focus": "focus_alternative_study",
    }
    return {
        "manifest_version": "1.0",
        "dataset_family": family,
        "dataset": datasets[family],
        "condition": condition,
        "item_id": item_id,
        "group_id": group_id,
        "context_id": item_id,
        "generation_frame": "no_frame",
        "generation_prompt": prompt,
        "trigger": trigger,
        "query": query,
        "analysis_inclusion_status": (
            "pending_hu_exact_filter_suite_available"
            if family == "hu_2023_benchmark"
            else "included"
        ),
        "has_hu_test_suite": True if family == "hu_2023_benchmark" else np.nan,
        "source_prompt_file": "source.csv",
        "source_row_id": item_id,
        "prompt_provenance": "test_fixture",
    }


def _valid_manifest() -> pd.DataFrame:
    rows = [
        _row(
            family="hu_2023_benchmark",
            condition="scalar_inference",
            item_id="hu-1",
            group_id="scale-1",
            prompt="It is ",
        ),
        _row(
            family="ronai_xiang_2024",
            condition="ESI",
            item_id="ESI::01",
            group_id="01",
            prompt="Story one.\nAnswer: ",
        ),
        _row(
            family="ronai_xiang_2024",
            condition="Eonly",
            item_id="Eonly::01",
            group_id="01",
            prompt="Story one.\nAnswer: ",
        ),
        _row(
            family="ronai_xiang_2024",
            condition="Eweak",
            item_id="Eweak::01",
            group_id="01",
            prompt="Story one.\nQuestion about warm?\nAnswer: ",
        ),
        _row(
            family="ronai_xiang_2024",
            condition="Estrong",
            item_id="Estrong::01",
            group_id="01",
            prompt="Story one.\nQuestion about hot?\nAnswer: ",
        ),
        _row(
            family="ronai_xiang_2024",
            condition="Eonlystrong",
            item_id="Eonlystrong::01",
            group_id="01",
            prompt="Story one.\nQuestion about hot?\nAnswer: ",
        ),
    ]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


class ScoringManifestTests(unittest.TestCase):
    def test_valid_manifest_and_summary(self) -> None:
        manifest = _valid_manifest()

        validate_scoring_manifest(manifest)
        summary = summarize_scoring_manifest(manifest)

        self.assertEqual(summary.row_count, 6)
        self.assertEqual(summary.unique_prompt_count, 4)
        self.assertEqual(summary.unique_prompt_candidate_count, 8)
        self.assertEqual(summary.hu_rows_pending_exact_filter, 1)

    def test_only_is_forbidden_in_rnx_answer_frame(self) -> None:
        manifest = _valid_manifest()
        manifest.loc[manifest["condition"].eq("Eonly"), "generation_prompt"] = (
            "Story one.\nAnswer: only "
        )

        with self.assertRaisesRegex(ValueError, "must not contain the word 'only'"):
            validate_scoring_manifest(manifest)

    def test_only_in_story_text_is_allowed(self) -> None:
        manifest = _valid_manifest()
        selected = manifest["condition"].isin(["ESI", "Eonly"])
        manifest.loc[selected, "generation_prompt"] = (
            "Only Pat attended the meeting.\nAnswer: "
        )

        validate_scoring_manifest(manifest)

    def test_only_condition_must_reuse_esi_prompt_and_candidates(self) -> None:
        manifest = _valid_manifest()
        manifest.loc[manifest["condition"].eq("Eonly"), "query"] = "boiling"

        with self.assertRaisesRegex(ValueError, "must match exactly"):
            validate_scoring_manifest(manifest)

    def test_prompt_must_end_in_whitespace(self) -> None:
        manifest = _valid_manifest()
        manifest.loc[manifest["condition"].eq("Eweak"), "generation_prompt"] = "Answer:"

        with self.assertRaisesRegex(ValueError, "must end in whitespace"):
            validate_scoring_manifest(manifest)

    def test_frame_aware_manifest_scores_query_only_after_x_but_not_y(self) -> None:
        row = _row(
            family="novel_focus",
            condition="focus_only",
            item_id="focus-1::x_but_not_y",
            group_id="story-1",
            prompt="It is warm but not ",
        )
        row.update(
            {
                "manifest_version": FRAME_AWARE_MANIFEST_VERSION,
                "generation_frame": "x_but_not_y",
                "score_trigger": False,
                "score_query": True,
            }
        )
        manifest = pd.DataFrame(
            [row],
            columns=[*MANIFEST_COLUMNS, *SCORING_CONTROL_COLUMNS],
        )

        validate_scoring_manifest(manifest)
        summary = summarize_scoring_manifest(manifest)

        self.assertEqual(summary.generation_frame_count, 1)
        self.assertEqual(summary.unique_prompt_candidate_count, 1)

    def test_x_but_not_y_cannot_score_trigger(self) -> None:
        row = _row(
            family="novel_focus",
            condition="focus_only",
            item_id="focus-1::x_but_not_y",
            group_id="story-1",
            prompt="It is warm but not ",
        )
        row.update(
            {
                "manifest_version": FRAME_AWARE_MANIFEST_VERSION,
                "generation_frame": "x_but_not_y",
                "score_trigger": True,
                "score_query": True,
            }
        )
        manifest = pd.DataFrame(
            [row],
            columns=[*MANIFEST_COLUMNS, *SCORING_CONTROL_COLUMNS],
        )

        with self.assertRaisesRegex(ValueError, "must score the query only"):
            validate_scoring_manifest(manifest)

    def test_rnx_cannot_use_x_but_not_y_frame(self) -> None:
        row = _row(
            family="ronai_xiang_2024",
            condition="ESI",
            item_id="ESI::01::x_but_not_y",
            group_id="01",
            prompt="It is warm but not ",
        )
        row.update(
            {
                "manifest_version": FRAME_AWARE_MANIFEST_VERSION,
                "generation_frame": "x_but_not_y",
                "score_trigger": False,
                "score_query": True,
            }
        )
        manifest = pd.DataFrame(
            [row],
            columns=[*MANIFEST_COLUMNS, *SCORING_CONTROL_COLUMNS],
        )

        with self.assertRaisesRegex(ValueError, "applies only to Hu and novel-focus"):
            validate_scoring_manifest(manifest)

    def test_committed_manifest_has_expected_coverage(self) -> None:
        path = PIPELINE_DIR / "scoring_manifests" / "hu_rnx_no_frame_manifest.csv"
        manifest = pd.read_csv(path)

        validate_scoring_manifest(manifest)
        counts = manifest.groupby("dataset_family").size().to_dict()
        rnx_counts = (
            manifest.loc[manifest["dataset_family"].eq("ronai_xiang_2024")]
            .groupby("condition")
            .size()
            .to_dict()
        )

        self.assertEqual(len(manifest), 609)
        self.assertEqual(counts, {"hu_2023_benchmark": 309, "ronai_xiang_2024": 300})
        self.assertEqual(
            rnx_counts,
            {"ESI": 60, "Eonly": 60, "Eonlystrong": 60, "Estrong": 60, "Eweak": 60},
        )

    def test_remaining_manifest_has_expected_frames_and_candidate_count(self) -> None:
        path = PIPELINE_DIR / "scoring_manifests" / "focus_hu_remaining_qwen_manifest.csv"
        manifest = pd.read_csv(path)

        validate_scoring_manifest(manifest)
        summary = summarize_scoring_manifest(manifest)
        counts = manifest.groupby(["dataset_family", "generation_frame"]).size().to_dict()

        self.assertEqual(len(manifest), 1269)
        self.assertEqual(
            counts,
            {
                ("hu_2023_benchmark", "x_but_not_y"): 309,
                ("novel_focus", "no_frame"): 480,
                ("novel_focus", "x_but_not_y"): 480,
            },
        )
        self.assertEqual(summary.unique_prompt_candidate_count, 871)


if __name__ == "__main__":
    unittest.main()
