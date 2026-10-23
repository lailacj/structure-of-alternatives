"""R&X frame policy, scoring contract, and extension parity; no Qwen inference."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "code"))

from rnx_contrastive import (DEFAULT_SOURCE, DEFAULT_MANIFEST, REVISION, FAMILY, FRAME_COLUMNS,
                            build_manifest, validate_scores, attach_scores, sha256)
from score_qwen_scoring_manifest import OUTPUT_COLUMNS, SCORING_BOUNDARY_VERSION, scoring_row_key
import score_qwen_scoring_manifest as scorer
from integrate_rnx_x_but_not_y import integrate
from build_linking_structure_tables import build_dataset_spearman
from canonical_observations import validate_canonical_observations


def fixture_scores(manifest):
    """Synthetic token scores confined to temporary test files, never research artifacts."""
    rows = []
    for _, row in manifest.iterrows():
        record = {column: np.nan for column in OUTPUT_COLUMNS}
        record.update({column: row[column] for column in scorer.MANIFEST_COLUMNS})
        # Equal prompts have equal scores; vary items/QUDs to exercise correlations.
        value = -1.0 - (sum(map(ord, row.generation_prompt)) % 93) / 10
        record.update(manifest_sha256=sha256(DEFAULT_MANIFEST), model_revision=REVISION,
                      model_identifier="Qwen/Qwen2-7B", model_revision_source="snapshot_directory",
                      resolved_model_path=f"/cluster/snapshots/{REVISION}", actual_model_dtype="torch.bfloat16",
                      torch_version="test", transformers_version="test", scorer_version=scorer.SCORER_VERSION,
                      scored_candidate_roles="query_only", scoring_boundary_version=SCORING_BOUNDARY_VERSION,
                      scoring_row_key=scoring_row_key(row), query_tokenization_mode="exact_concat",
                      query_token_count=2, query_logprob_sum=value, query_logprob_mean=value/2,
                      query_token_ids_json="[1, 2]", query_tokens_json='["test1", "test2"]',
                      query_token_logprobs_json=json.dumps([value/2, value/2]))
        rows.append(record)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


class RnxContrastiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = pd.read_csv(DEFAULT_SOURCE)
        cls.manifest = pd.read_csv(DEFAULT_MANIFEST, dtype={"manifest_version": str})
        cls.scores = fixture_scores(cls.manifest)

    def validate(self, scores):
        validate_scores(scores, self.manifest, manifest_sha256=sha256(DEFAULT_MANIFEST))

    def test_committed_manifest_is_reproducible_and_matches_movie_policy(self):
        rebuilt = build_manifest(self.source, source_file=self.manifest.source_prompt_file.iloc[0])
        pd.testing.assert_frame_equal(rebuilt, self.manifest)
        self.assertEqual(len(rebuilt), 300)
        self.assertEqual(len(rebuilt[["generation_prompt", "query"]].drop_duplicates()), 180)
        movie = rebuilt.loc[rebuilt.generation_prompt.str.contains("movie") & rebuilt.trigger.eq("good")]
        expected = {
            "ESI": "The movie is good but not ", "Eonly": "The movie is good but not ",
            "Eweak": "Sue: Is the movie good?\nMary: The movie is good but not ",
            "Estrong": "Sue: Is the movie excellent?\nMary: The movie is good but not ",
            "Eonlystrong": "Sue: Is the movie excellent?\nMary: The movie is good but not ",
        }
        self.assertEqual(movie.set_index("condition").generation_prompt.to_dict(), expected)
        self.assertTrue(movie["query"].eq("excellent").all())
        self.assertTrue(rebuilt.score_query.all())
        self.assertFalse(rebuilt.score_trigger.any())

    def test_incomplete_or_misaligned_manifest_fails(self):
        with self.assertRaises(ValueError):
            build_manifest(self.source.drop(self.source.index[self.source.dataset_family.eq(FAMILY)][0]), source_file="test")
        changed = self.source.copy()
        changed.loc[changed.condition.eq("Eonly"), "trigger"] = "changed"
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            build_manifest(changed, source_file="test")

    def test_scored_artifact_contract_and_failure_cases(self):
        self.validate(self.scores)
        cases = [("model_revision", "other"), ("manifest_sha256", "other"),
                 ("generation_prompt", "It is wrong but not "), ("query", "wrong"),
                 ("scored_candidate_roles", "trigger_and_query"), ("query_logprob_sum", np.inf),
                 ("scoring_boundary_version", "legacy"), ("query_tokenization_mode", "fallback"),
                 ("trigger_logprob_sum", -2), ("query_token_count", 1),
                 ("query_token_logprobs_json", "[-99, -99]"), ("scoring_row_key", "other")]
        for column, value in cases:
            with self.subTest(column=column):
                changed = self.scores.copy()
                changed.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.validate(changed)
        for changed in [self.scores.iloc[:-1], pd.concat([self.scores.iloc[:-1], self.scores.iloc[[0]]])]:
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_conflicting_shared_prompt_scores_are_rejected(self):
        changed = self.scores.copy()
        changed.loc[0, "query_token_logprobs_json"] = "[-1, -2]"
        changed.loc[0, "query_logprob_sum"] = -3
        changed.loc[0, "query_logprob_mean"] = -1.5
        with self.assertRaisesRegex(ValueError, "conflicting scores"):
            self.validate(changed)

    def test_attachment_preserves_neutral_scores_human_data_and_other_datasets(self):
        attached = attach_scores(self.source, self.scores, artifact="test-only")
        preserved = [column for column in self.source if column not in FRAME_COLUMNS]
        pd.testing.assert_frame_equal(self.source[preserved], attached[preserved])
        validate_canonical_observations(attached, require_complete_provenance=True)
        rnx = attached.loc[attached.dataset_family.eq(FAMILY)]
        self.assertTrue(rnx.x_but_not_y_applicable.all())
        for a, b in [("ESI", "Eonly"), ("Estrong", "Eonlystrong")]:
            x = rnx.loc[rnx.condition.eq(a)].set_index("group_id").x_but_not_y_logprob_sum
            y = rnx.loc[rnx.condition.eq(b)].set_index("group_id").x_but_not_y_logprob_sum
            pd.testing.assert_series_equal(x, y)
        changed = self.source.copy()
        changed.loc[changed.condition.eq("Eweak"), "generation_prompt"] += "Changed "
        with self.assertRaisesRegex(ValueError, "parent neutral inputs"):
            attach_scores(changed, self.scores, artifact="test-only")

    def test_dry_run_never_loads_model_or_writes_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.csv"
            args = SimpleNamespace(manifest=DEFAULT_MANIFEST, model_path=f"/unused/snapshots/{REVISION}",
                                   output=output, dry_run=True)
            with mock.patch.object(scorer, "load_model", side_effect=AssertionError("No inference allowed")), contextlib.redirect_stdout(io.StringIO()):
                scorer.score_manifest(args)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_new_bundle_preserves_sampled_predictions_and_exposes_60_framed_items_per_condition(self):
        parent = DEFAULT_SOURCE.parent.parent
        parent_hashes = {p: sha256(p) for p in [DEFAULT_SOURCE, parent / "prediction_grid.csv", parent / "cv_results/oof_predictions.csv"]}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scores = directory / "synthetic_scores.csv"
            self.scores.to_csv(scores, index=False)
            output = directory / "test_bundle"
            with contextlib.redirect_stdout(io.StringIO()):
                integrate(parent_results=parent, manifest_path=DEFAULT_MANIFEST, scores_path=scores, output=output)
            attached = pd.read_csv(output / "inputs/source_rows.csv")
            oof = pd.read_csv(output / "cv_results/oof_predictions.csv")
            _, summary, _ = build_dataset_spearman(attached, oof)
            framed = summary.loc[summary.analysis_dataset_id.str.startswith("rnx_") & summary.model.eq("X but not Y")]
            self.assertEqual(len(framed), 5)
            self.assertTrue(framed.n.eq(60).all())
            for relative in ["prediction_grid.csv", "cv_results/oof_predictions.csv"]:
                old = pd.read_csv(parent / relative).drop(columns="source_rows_sha256")
                new = pd.read_csv(output / relative).drop(columns="source_rows_sha256")
                pd.testing.assert_frame_equal(old, new, check_exact=True)
            self.assertEqual(sha256(parent / "cv_results/fold_selections.csv"), sha256(output / "cv_results/fold_selections.csv"))
            # Exercise table generation and verified parent-distribution reuse without inference.
            subprocess.run([sys.executable, str(PIPELINE / "code/build_linking_structure_tables.py"),
                            "--source-rows", str(output / "inputs/source_rows.csv"), "--results-dir", str(output),
                            "--output-dir", str(output / "linking_structure_tables")], check=True, capture_output=True)
            import build_results_viewer as viewer
            payload = viewer.build_payload(output, output / "inputs")
            parent_viewer = PIPELINE / "results_viewer/index.html"
            self.assertEqual(viewer.preserve_distributions(payload, parent_viewer), 360)
            self.assertEqual(payload["verifiedCells"], 261)
            self.assertTrue(all(payload["datasetSummaries"][f"rnx_{c}"][1]["n"] == 60
                                for c in ["esi", "eweak", "estrong", "eonly", "eonlystrong"]))
            changed = copy.deepcopy(payload)
            changed["rnxContrastiveExtension"]["parent_source_rows_sha256"] = "tampered"
            with self.assertRaisesRegex(ValueError, "source_rows.csv changed"):
                viewer.preserve_distributions(changed, parent_viewer)
            # Original serialized values must survive exactly, even in neutral floating-point fields.
            with DEFAULT_SOURCE.open(newline="") as f:
                original = list(csv.DictReader(f))
            with (output / "inputs/source_rows.csv").open(newline="") as f:
                extended = list(csv.DictReader(f))
            for a, b in zip(original, extended):
                for field in a:
                    if field not in FRAME_COLUMNS or a["dataset_family"] != FAMILY:
                        self.assertEqual(a[field], b[field])
            with self.assertRaises(FileExistsError):
                integrate(parent_results=parent, manifest_path=DEFAULT_MANIFEST, scores_path=scores, output=output)
        for path, digest in parent_hashes.items():
            self.assertEqual(sha256(path), digest)

    def test_optional_canonical_rebuild_joins_the_same_condition_item_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scores = directory / "synthetic_scores.csv"
            self.scores.to_csv(scores, index=False)
            subprocess.run([sys.executable, str(PIPELINE / "code/build_cross_dataset_canonical_observations.py"),
                            "--rnx-x-frame-scores", str(scores), "--output-dir", str(directory / "canonical")],
                           check=True, capture_output=True)
            rnx = pd.read_csv(directory / "canonical/ronai_xiang_2024_observations.csv")
            self.assertTrue(rnx.x_but_not_y_applicable.all())
            self.assertEqual(len(rnx), 300)
            validate_canonical_observations(rnx, require_complete_provenance=True, require_human_counts=True)


if __name__ == "__main__":
    unittest.main()
