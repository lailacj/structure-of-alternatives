"""Checks for matched neutral artifacts and the historical framed audit."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from export_corrected_set_variant_sources import REVISION, audit_framed, export_neutral, sha
from score_qwen_scoring_manifest import SCORING_BOUNDARY_VERSION
from build_set_variant_prediction_grid import validate_corrected_array_identity


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(text.encode()))

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]


class CorrectedScoreTests(unittest.TestCase):
    def fixture(self, directory):
        unigram = directory / "unigram.txt"
        bigram = directory / "bigram.txt"
        unigram.write_text("warm\nhot\n")
        bigram.write_text("very hot\n")
        (directory / "vocab_manifest.json").write_text(json.dumps({"total_count": 3, "sources": [
            {"name": "1gram", "path": str(unigram), "offset": 0, "count": 2},
            {"name": "2gram", "path": str(bigram), "offset": 2, "count": 1}]}))
        path = directory / "prompt_a.log_probs.npy"
        np.save(path, np.array([-1., -2., -3.], dtype=np.float32))
        meta = {"context": "prompt_a", "prompt": "It is ", "scoring_prefix": "It is", "continuation_prefix": " ",
                "scoring_boundary_version": SCORING_BOUNDARY_VERSION, "target_vocab_size": 3,
                "model_path": "/cache/snapshots/" + REVISION}
        (directory / "prompt_a.meta.json").write_text(json.dumps(meta))
        (directory / "prompt_a.progress.json").write_text(json.dumps({"sources": {"1gram": {"done": True}, "2gram": {"done": True}}}))
        source = pd.DataFrame([{"prompt_id": "prompt_a", "generation_prompt": "It is ", "trigger": "warm", "query": "hot",
            "model_revision": REVISION, "human_rate": .7, "human_yes": 7, "human_total": 10,
            "source_score_file": "historical.csv", "trigger_logprob_sum": -9., "query_logprob_sum": -8.,
            "trigger_logprob_mean": -9., "query_logprob_mean": -8., "trigger_token_count": 1, "query_token_count": 1,
            "trigger_tokenization_mode": "exact_concat", "query_tokenization_mode": "exact_concat",
            "x_but_not_y_logprob_sum": -.5}])
        prompts = pd.DataFrame([{"prompt_id": "prompt_a", "generation_prompt": "It is "}])
        return source, prompts, path

    def test_export_uses_array_values_and_preserves_humans_and_framed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, prompts, path = self.fixture(directory)
            before = sha(path)
            corrected, targets, provenance = export_neutral(source, prompts, directory, Tokenizer())
            self.assertEqual(sha(path), before)
            self.assertEqual(source.query_logprob_sum.iloc[0], -8.)
            self.assertEqual(corrected.query_logprob_sum.iloc[0], -2.)
            self.assertEqual(corrected.trigger_logprob_sum.iloc[0], -1.)
            self.assertEqual(corrected.query_token_count.iloc[0], 4)
            self.assertEqual(corrected.query_logprob_mean.iloc[0], -.5)
            self.assertEqual(corrected.human_rate.iloc[0], .7)
            self.assertEqual(corrected.x_but_not_y_logprob_sum.iloc[0], -.5)
            self.assertEqual(len(targets), 2)
            self.assertEqual(len(provenance), 1)
            self.assertAlmostEqual(corrected.query_sampling_probability.iloc[0], np.exp(-2) / np.exp([-1., -2., -3.]).sum())
            validate_corrected_array_identity(corrected, path, np.load(path), {"warm": 0, "hot": 1})
            bad = corrected.copy()
            bad.loc[0, "query_logprob_sum"] = -8
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_corrected_array_identity(bad, path, np.load(path), {"warm": 0, "hot": 1})
            np.save(path, np.array([-1., -2., -4.], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "different hashes"):
                validate_corrected_array_identity(corrected, path, np.load(path), {"warm": 0, "hot": 1})

    def test_legacy_or_changed_array_metadata_is_rejected(self):
        for field, value in [("scoring_boundary_version", "legacy"), ("prompt", "Changed "),
                             ("continuation_prefix", "  "), ("model_path", "/wrong/snapshot")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source, prompts, _ = self.fixture(directory)
                path = directory / "prompt_a.meta.json"
                meta = json.loads(path.read_text())
                meta[field] = value
                path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    export_neutral(source, prompts, directory, Tokenizer())

    def test_incomplete_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, prompts, _ = self.fixture(directory)
            (directory / "prompt_a.progress.json").write_text(json.dumps({"sources": {"1gram": {"done": True}, "2gram": {"done": False}}}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export_neutral(source, prompts, directory, Tokenizer())

    def framed_fixture(self):
        ids = list(b" hot")
        frame = pd.DataFrame([{"generation_frame": "x_but_not_y", "generation_prompt": "It is warm but not ", "query": "hot",
            "model_revision": REVISION, "query_tokenization_mode": "exact_concat", "scored_candidate_roles": "query_only",
            "trigger_logprob_sum": np.nan, "query_token_ids_json": json.dumps(ids), "query_token_count": len(ids),
            "query_token_logprobs_json": json.dumps([-1.]*len(ids)), "query_tokens_json": json.dumps([chr(i) for i in ids]),
            "query_logprob_sum": -4., "query_logprob_mean": -1.}])
        source = pd.DataFrame([{"query": "hot", "x_but_not_y_prompt": "It is warm but not ",
            "x_but_not_y_logprob_sum": -4., "x_but_not_y_logprob_mean": -1., "x_but_not_y_token_count": 4}])
        return source, frame

    def test_existing_framed_scores_pass_without_relabelling(self):
        source, frame = self.framed_fixture()
        before = frame.copy(deep=True)
        report = audit_framed(source, frame, Tokenizer())
        self.assertTrue(report["passed"])
        self.assertEqual(report["unique_prompt_candidate_pairs"], 1)
        pd.testing.assert_frame_equal(before, frame)

    def test_framed_boundary_tokenization_scores_and_revision_are_checked(self):
        for field, value in [("generation_prompt", "It is warm but not  "), ("query_token_ids_json", "[1,2,3,4]"),
                             ("query_logprob_sum", -99.), ("model_revision", "wrong")]:
            with self.subTest(field=field):
                source, frame = self.framed_fixture()
                frame.loc[0, field] = value
                with self.assertRaises(ValueError):
                    audit_framed(source, frame, Tokenizer())


if __name__ == "__main__":
    unittest.main()
