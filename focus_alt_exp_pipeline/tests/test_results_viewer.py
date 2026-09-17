"""Numerical parity, source joins, and optional full-distribution export checks."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "code"))
import build_results_viewer as viewer
import viewer_rank_association as association


class ResultsViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = viewer.build_payload(PIPELINE / "results/set_variant_qwen", PIPELINE / "scoring_manifests/set_variant_qwen")

    def test_all_saved_metric_cells_and_item_scores_match(self):
        self.assertEqual(self.payload["verifiedCells"], 261)
        self.assertEqual(len(self.payload["items"]), 993)
        saved = viewer.read_csv(PIPELINE / "results/set_variant_qwen/advisor_summary/item_level_scatterplots/item_level_predictions_and_log_scores.csv")
        lookup = {(r["dataset"], r["id"]): r for r in self.payload["items"]}
        self.assertEqual(len(saved), 993 * 9 - 300)
        for row in saved:
            rebuilt = lookup[row["analysis_dataset_id"], row["analysis_unit_id"]]
            index = viewer.MODELS.index(row["model"])
            self.assertAlmostEqual(rebuilt["p"][index], float(row["prediction"]), places=10)
            self.assertAlmostEqual(rebuilt["scores"][index], float(row["item_log_score"]), places=8)

    def test_coverage_and_template_grain(self):
        self.assertEqual(len(self.payload["contexts"]), 16)
        for context in self.payload["contexts"]:
            self.assertEqual(self.payload["contextSummaries"][context][0]["n"], 30)
        self.assertEqual(self.payload["datasetSummaries"]["hu_vt16"][0]["n"], 39)
        vt = [r for r in self.payload["items"] if r["dataset"] == "hu_vt16"]
        self.assertTrue(all(len(row["sources"]) == 3 for row in vt))
        for dataset, _ in viewer.DATASETS:
            if dataset.startswith("rnx_"):
                stats = self.payload["datasetSummaries"][dataset][1]
                self.assertEqual(stats["n"], 0)
                self.assertIsNone(stats["r"])
                self.assertIsNone(stats["log"])

    def test_statistic_edge_cases(self):
        self.assertIsNone(viewer.pearson([1, 1], [0, 1]))
        self.assertIsNone(viewer.pearson([1], [0]))
        self.assertAlmostEqual(viewer.pearson([0, 1, 2], [2, 1, 0]), -1)
        self.assertAlmostEqual(viewer.log_score(.5, .5), math.log(.5))
        self.assertTrue(math.isfinite(viewer.log_score(1, 0)))
        with self.assertRaises(ValueError):
            viewer.log_score(.5, 1.1)

    def test_rebuild_preserves_distributions_but_rejects_changed_inputs(self):
        old = copy.deepcopy(self.payload)
        prompt = next(p for p in old["prompts"] if p["frame"] == "Neutral" and "bag" in p["contexts"])
        prompt["distribution"] = [{"word": "test", "logp": -2., "rank": 1, "normalized": .1}]
        prompt["distributionCoverage"] = "Top 50 + experimental alternatives from full support"
        prompt["supportSize"] = 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previous.html"
            path.write_text(viewer.render_html(old, PIPELINE / "results_viewer"), encoding="utf-8")
            rebuilt = copy.deepcopy(self.payload)
            self.assertEqual(viewer.preserve_distributions(rebuilt, path), 2)
            recovered = next(p for p in rebuilt["prompts"] if p["id"] == prompt["id"])
            self.assertEqual(recovered["distribution"], prompt["distribution"])
            self.assertEqual(rebuilt["items"], self.payload["items"])
            changed = copy.deepcopy(self.payload)
            next(p for p in changed["provenance"] if p["path"].endswith("source_rows.csv"))["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "source_rows.csv changed"):
                viewer.preserve_distributions(changed, path)
            changed = copy.deepcopy(self.payload)
            next(p for p in changed["prompts"] if p["id"] == prompt["id"])["text"] = "different prompt"
            with self.assertRaisesRegex(ValueError, "changed prompt"):
                viewer.preserve_distributions(changed, path)

    def test_missing_variant_is_rejected(self):
        source = viewer.read_csv(PIPELINE / "scoring_manifests/set_variant_qwen/source_rows.csv")
        oof = viewer.read_csv(PIPELINE / "results/set_variant_qwen/cv_results/oof_predictions.csv")
        with self.assertRaisesRegex(ValueError, "Incomplete variants"):
            viewer.build_items(source, oof[1:])

    def test_every_item_links_to_exact_prompts(self):
        prompts = {p["id"]: p for p in self.payload["prompts"]}
        self.assertEqual(sum(p["frame"] == "Neutral" for p in prompts.values()), 360)
        for item in self.payload["items"]:
            for index in item["sources"]:
                row = self.payload["sources"][index]
                self.assertTrue(row["included"])
                self.assertIn(row["prompt"], prompts)
                if row["framedPrompt"]:
                    self.assertEqual(prompts[row["framedPrompt"]]["frame"], "X but not Y")
        matched = [p for p in prompts.values() if "rnx_esi" in p["datasets"]]
        self.assertTrue(all("rnx_eonly" in p["datasets"] for p in matched))

    def test_raw_and_normalized_probabilities_are_distinct(self):
        prompt = next(p for p in self.payload["prompts"] if p["frame"] == "Neutral" and "mask" in p["contexts"])
        candidate = prompt["distribution"][0]
        self.assertEqual(candidate["word"], "mask")
        self.assertAlmostEqual(candidate["normalized"], .6522432897840695)
        self.assertAlmostEqual(math.exp(candidate["logp"]), .012267934489661783)

    def test_html_embeds_safe_json_without_network_dependencies(self):
        payload = {"test": "</script><script>alert('x')</script>\u2028"}
        html = viewer.render_html(payload, PIPELINE / "results_viewer")
        data = html.split('<script id="results-data" type="application/json">')[1].split("</script>")[0]
        self.assertEqual(json.loads(data), payload)
        self.assertNotIn("<script src=", html)
        self.assertNotIn("__VIEWER_DATA__", html)


class FullDistributionTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Full-array export tests require optional NumPy")
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory / "vocab.txt").write_text("alpha\nbeta\ngamma\ntarget\n")
        (self.directory / "vocab_manifest.json").write_text(json.dumps({"total_count": 4, "sources": [{"path": "/cluster/vocab.txt", "count": 4}]}))
        np.save(self.directory / "prompt_test.log_probs.npy", np.log([.4, .2, .1, .01]).astype(np.float32))
        self.metadata = {"context": "prompt_test", "target_vocab_size": 4, "prompt": "The next word is "}
        (self.directory / "prompt_test.meta.json").write_text(json.dumps(self.metadata))
        (self.directory / "prompt_test.progress.json").write_text(json.dumps({"sources": {"1gram": {"done": True}, "2gram": {"done": True}}}))
        self.prompts = {"prompt_test": {"id": "prompt_test", "text": "The next word is ", "frame": "Neutral", "candidates": [{"word": "target"}]}}

    def test_top_export_keeps_targets_and_normalizes_full_support(self):
        viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory)
        rows = self.prompts["prompt_test"]["distribution"]
        self.assertEqual([r["word"] for r in rows], ["alpha", "beta", "target"])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 4])
        self.assertAlmostEqual(rows[0]["normalized"], .4/.71, places=7)
        self.assertLess(sum(r["normalized"] for r in rows), 1)

    def test_framed_scores_match_prompt_text_despite_different_job_id(self):
        prompt = self.prompts.pop("prompt_test")
        prompt.update(id="framed_viewer_id", frame="X but not Y")
        self.prompts[prompt["id"]] = prompt
        viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory, frame="X but not Y")
        self.assertEqual([r["word"] for r in prompt["distribution"]], ["alpha", "beta", "target"])
        self.assertAlmostEqual(prompt["distribution"][0]["normalized"], .4/.71, places=7)

    def test_framed_missing_or_ambiguous_prompt_is_rejected(self):
        prompt = self.prompts["prompt_test"]
        prompt["frame"] = "X but not Y"
        prompt["text"] = "not the scored prompt"
        with self.assertRaisesRegex(ValueError, "0 exact prompt matches"):
            viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory, frame="X but not Y")
        prompt["text"] = self.metadata["prompt"]
        (self.directory / "duplicate.meta.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "2 exact prompt matches"):
            viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory, frame="X but not Y")

    def test_all_export_and_bad_prompt(self):
        viewer.add_full_distributions(self.prompts, self.directory, 0, self.directory)
        self.assertAlmostEqual(sum(r["normalized"] for r in self.prompts["prompt_test"]["distribution"]), 1)
        self.metadata["prompt"] = "Wrong prompt "
        (self.directory / "prompt_test.meta.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "prompt text mismatch"):
            viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory)

    def test_incomplete_or_nonfinite_array_is_rejected(self):
        self.np.save(self.directory / "prompt_test.log_probs.npy", self.np.array([0, -1, -2, self.np.nan]))
        with self.assertRaisesRegex(ValueError, "Invalid score array"):
            viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory)

    def test_corrected_array_hash_is_checked_before_export(self):
        self.prompts["prompt_test"]["expectedArraySha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Array hash disagrees"):
            viewer.add_full_distributions(self.prompts, self.directory, 2, self.directory)


class RankAssociationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = viewer.build_payload(PIPELINE / "results/set_variant_qwen", PIPELINE / "scoring_manifests/set_variant_qwen")
        viewer.preserve_distributions(cls.payload, PIPELINE / "results_viewer/index.html")
        cls.original = copy.deepcopy(cls.payload)
        association.add_rank_analysis(cls.payload)

    def test_recomputed_tests_reproduce_independent_exploratory_analysis(self):
        expected = viewer.read_csv(PIPELINE / "diagnostics/context_rank_association_2026-09-17/association_tests.csv")
        self.assertEqual(len(self.payload["rankAssociation"]["tests"]), 18)
        for row in self.payload["rankAssociation"]["tests"]:
            model = self.payload["models"][row["model"]]
            label = "No linking" if model == "No linking structure" else model
            definition = "sampling_word_rho" if row["definition"] == "sampling" else "viewer_word_rho"
            old = next(r for r in expected if r["word_definition"] == definition and r["predictor"] == label)
            for key, column in (("R", "spearman"), ("p", "permutation_p_two_sided"),
                                ("holm9", "holm_p_within_definition_9"),
                                ("holm18", "holm_p_all_18_sensitivity"),
                                ("ciLow", "bootstrap_95_low"), ("ciHigh", "bootstrap_95_high")):
                self.assertAlmostEqual(row[key], float(old[column]), places=10)

    def test_all_scores_and_predictions_have_matching_provenance(self):
        self.assertEqual(len(self.payload["spearman"]["word_paired_ranks"]), 96)
        for row in self.payload["spearman"]["word_paired_ranks"]:
            prompt = next(p for p in self.payload["prompts"] if p["id"] == row["prompt_id"])
            candidate = next(c for c in prompt["distribution"] if c["word"] == row["word"])
            self.assertEqual(candidate["logp"], row["model_value"])
        for key in ("items", "datasetSummaries", "contextSummaries", "prompts"):
            self.assertEqual(self.payload[key], self.original[key])
        self.assertEqual(self.payload["spearman"]["negation_paired_ranks"], self.original["spearman"]["negation_paired_ranks"])

    def test_missing_sampling_scores_are_not_replaced_with_direct_scores(self):
        payload = copy.deepcopy(self.original)
        prompt = next(p for p in payload["prompts"] if "fridge" in p["contexts"] and p["frame"] == "Neutral")
        prompt["distribution"] = []
        association.add_rank_analysis(payload, permutations=99, bootstraps=99)
        row = next(r for r in payload["spearman"]["word_spearman_by_context"] if r["context"] == "fridge")
        self.assertIsNone(row["spearman_rho"])
        self.assertEqual(row["status"], "missing_sampling_scores")
        for test in payload["rankAssociation"]["tests"]:
            if test["definition"] == "sampling":
                self.assertIn("fridge", test["omitted"])

    def test_corrected_run_refuses_mixed_target_and_sampling_artifacts(self):
        payload = copy.deepcopy(self.original)
        payload["run"]["corrected"] = True
        with self.assertRaisesRegex(ValueError, "requires matching direct-target"):
            association.add_rank_analysis(payload, permutations=9, bootstraps=9)

    def test_rank_ties_and_undefined_associations(self):
        self.assertAlmostEqual(association.rho([1, 1, 3, 4], [4, 3, 2, 1]), -.9486832980505138)
        self.assertIsNone(association.rho([1, 1, 1], [1, 2, 3]))
        self.assertEqual(association.holm([.01, .04, .03]), [.03, .06, .06])


if __name__ == "__main__":
    unittest.main()
