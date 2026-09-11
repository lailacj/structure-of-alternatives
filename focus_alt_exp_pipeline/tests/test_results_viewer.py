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


class ResultsViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = viewer.build_payload(PIPELINE / "results/set_variant_qwen", PIPELINE / "scoring_manifests/set_variant_qwen")

    def test_all_saved_metric_cells_and_item_scores_match(self):
        self.assertEqual(self.payload["verifiedCells"], 180)
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


if __name__ == "__main__":
    unittest.main()
