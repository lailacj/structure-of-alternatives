"""Tests for pure utilities in the resumable Qwen manifest scorer."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import score_qwen_scoring_manifest as scorer  # noqa: E402
from score_qwen_scoring_manifest import (  # noqa: E402
    ContinuationScore,
    OUTPUT_COLUMNS,
    continuation_token_ids,
    partial_output_path,
    resolve_model_location,
    score_manifest,
    scoring_row_key,
    split_prompt_boundary,
)
from scoring_manifest import MANIFEST_COLUMNS  # noqa: E402


class CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(character) for character in text])


class BoundaryMergingTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        if text == "It is":
            return SimpleNamespace(input_ids=[1, 2])
        return SimpleNamespace(input_ids=[1, 9])


class QwenManifestScorerTests(unittest.TestCase):
    def test_prompt_whitespace_is_moved_to_continuation(self) -> None:
        prefix, continuation = split_prompt_boundary("Question?\nAnswer: ", " warm ")

        self.assertEqual(prefix, "Question?\nAnswer:")
        self.assertEqual(continuation, " warm")

    def test_exact_concat_token_ids_are_selected(self) -> None:
        tokenizer = CharacterTokenizer()

        token_ids, mode = continuation_token_ids(
            tokenizer,
            prefix="It is",
            continuation=" warm",
        )

        self.assertEqual(token_ids, [ord(character) for character in " warm"])
        self.assertEqual(mode, "exact_concat")

    def test_boundary_merging_is_rejected_instead_of_silently_approximated(self) -> None:
        with self.assertRaisesRegex(ValueError, "merged across"):
            continuation_token_ids(
                BoundaryMergingTokenizer(),
                prefix="It is",
                continuation=" warm",
            )

    def test_cache_root_resolves_ref_main_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "models--Qwen--Qwen2-7B"
            snapshot = cache / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (cache / "refs").mkdir()
            (cache / "refs" / "main").write_text("abc123\n", encoding="utf-8")

            location = resolve_model_location(cache)

            self.assertEqual(location.resolved, str(snapshot.resolve()))
            self.assertEqual(location.revision, "abc123")
            self.assertEqual(location.revision_source, "huggingface_cache_ref_main")

    def test_multiple_unreferenced_snapshots_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "models--Qwen--Qwen2-7B"
            (cache / "snapshots" / "abc123").mkdir(parents=True)
            (cache / "snapshots" / "def456").mkdir()

            with self.assertRaisesRegex(ValueError, "multiple snapshots"):
                resolve_model_location(cache)

    def test_scoring_key_is_stable_and_uses_manifest_identity(self) -> None:
        row = {
            "dataset_family": "ronai_xiang_2024",
            "dataset": "ronai_xiang_2024",
            "condition": "ESI",
            "item_id": "ESI::01",
        }

        self.assertEqual(
            scoring_row_key(row),
            '["ronai_xiang_2024","ronai_xiang_2024","ESI","ESI::01"]',
        )

    def test_partial_output_name_is_predictable(self) -> None:
        self.assertEqual(
            partial_output_path(Path("scores.csv")),
            Path("scores.partial.csv"),
        )

    def test_end_to_end_checkpoint_and_resume_without_model_load(self) -> None:
        manifest_row = {
            "manifest_version": "1.0",
            "dataset_family": "hu_2023_benchmark",
            "dataset": "toy_hu",
            "condition": "scalar_inference",
            "item_id": "hu-1",
            "group_id": "scale-1",
            "context_id": "hu-1",
            "generation_frame": "no_frame",
            "generation_prompt": "It is ",
            "trigger": "warm",
            "query": "hot",
            "analysis_inclusion_status": "pending_hu_exact_filter_suite_available",
            "has_hu_test_suite": True,
            "source_prompt_file": "source.csv",
            "source_row_id": "hu-1",
            "prompt_provenance": "test_fixture",
        }
        fake_metadata = {
            "model_identifier": "fake-qwen",
            "model_revision": "abc123",
            "model_revision_source": "test_fixture",
            "resolved_model_path": "/fake/snapshots/abc123",
            "tokenizer_name_or_path": "fake-qwen",
            "tokenizer_class": "FakeTokenizer",
            "model_class": "FakeModel",
            "requested_dtype": "bfloat16",
            "actual_model_dtype": "bfloat16",
            "device_map": "auto",
            "local_files_only": True,
            "torch_version": "test",
            "transformers_version": "test",
        }
        trigger_score = ContinuationScore(
            token_ids=(1,),
            tokens=("warm",),
            token_logprobs=(-2.0,),
            tokenization_mode="exact_concat",
        )
        query_score = ContinuationScore(
            token_ids=(2,),
            tokens=("hot",),
            token_logprobs=(-1.0,),
            tokenization_mode="exact_concat",
        )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = directory / "manifest.csv"
            output_path = directory / "scores.csv"
            pd.DataFrame([manifest_row], columns=MANIFEST_COLUMNS).to_csv(
                manifest_path,
                index=False,
            )
            args = SimpleNamespace(
                manifest=manifest_path,
                output=output_path,
                model_path="fake-qwen",
                dtype="bfloat16",
                device_map="auto",
                allow_downloads=False,
                checkpoint_every=1,
                resume=True,
                overwrite=False,
                dry_run=False,
            )

            with mock.patch.object(
                scorer,
                "load_model",
                return_value=(object(), object(), object(), fake_metadata),
            ), mock.patch.object(
                scorer,
                "_candidate_scores_for_prompt",
                return_value={"warm": trigger_score, "hot": query_score},
            ):
                score_manifest(args)

            output = pd.read_csv(output_path)
            self.assertEqual(list(output.columns), list(OUTPUT_COLUMNS))
            self.assertEqual(len(output), 1)
            self.assertEqual(output.loc[0, "model_revision"], "abc123")
            self.assertEqual(output.loc[0, "query_minus_trigger_logprob_sum"], 1.0)
            checkpoint = partial_output_path(output_path)
            self.assertFalse(checkpoint.exists())

            output_path.rename(checkpoint)
            with mock.patch.object(
                scorer,
                "load_model",
                return_value=(object(), object(), object(), fake_metadata),
            ), mock.patch.object(
                scorer,
                "_candidate_scores_for_prompt",
                side_effect=AssertionError("completed rows must not be rescored"),
            ):
                score_manifest(args)

            self.assertTrue(output_path.exists())
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
