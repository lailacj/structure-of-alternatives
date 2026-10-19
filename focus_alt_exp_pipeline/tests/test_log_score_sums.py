"""Reporting sums the unchanged item scores; fitting objectives retain means."""
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "code"))
import build_results_viewer as viewer
from evaluate_set_variant_grid import summarize_log_scores


class LogScoreSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15"
        cls.payload = viewer.build_payload(results, results / "inputs")

    def test_no_linking_total_and_every_context_partition(self):
        payload = self.payload
        self.assertEqual(payload["logScoreAggregation"], "sum_of_item_log_scores")
        self.assertAlmostEqual(payload["datasetSummaries"]["novel_focus"][0]["log"], -1544.387762, places=5)
        for model in range(len(viewer.MODELS)):
            context_total = math.fsum(payload["contextSummaries"][c][model]["log"] for c in payload["contexts"])
            self.assertAlmostEqual(context_total, payload["datasetSummaries"]["novel_focus"][model]["log"], places=9)

    def test_every_dataset_total_matches_independent_binary_log_score(self):
        payload = self.payload
        for dataset in payload["datasets"]:
            for model in range(len(viewer.MODELS)):
                rows = [r for r in payload["items"] if r["dataset"] == dataset["id"] and r["p"][model] is not None]
                if not rows:
                    self.assertIsNone(payload["datasetSummaries"][dataset["id"]][model]["log"])
                    continue
                scores = []
                for row in rows:
                    p = min(1 - 1e-10, max(1e-10, row["p"][model]))
                    scores.append(row["y"] * math.log(p) + (1 - row["y"]) * math.log1p(-p))
                self.assertAlmostEqual(math.fsum(scores), payload["datasetSummaries"][dataset["id"]][model]["log"], places=9)

    def test_saved_long_exports_match_all_dataset_and_context_sums(self):
        path = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15/linking_structure_tables/summed_log_scores.csv"
        rows = viewer.read_csv(path)
        self.assertEqual(len(rows), 234)  # 90 dataset/model slots + 144 context/model slots.
        for row in rows:
            model = viewer.MODELS.index(row["model"])
            stats = (self.payload["datasetSummaries"][row["dataset_id"]][model] if row["context"] == "all"
                     else self.payload["contextSummaries"][row["context"]][model])
            self.assertEqual(int(row["n"]), stats["n"])
            value = viewer.number(row["sum_log_score"])
            if stats["log"] is None:
                self.assertIsNone(value)
            else:
                self.assertAlmostEqual(value, stats["log"], places=9)

    def test_cv_reports_sums_without_replacing_original_balanced_means(self):
        records = []
        for dataset, n, p in [("a", 1, .5), ("b", 3, .25)]:
            for _ in range(n):
                records.append(dict(variant="top_k", analysis_dataset_id=dataset, analysis_label=dataset,
                                    human_rate=1., **{s + "_probability": p for s in ("set", "ordering", "conjunction", "disjunction")}))
        datasets, aggregate = summarize_log_scores(pd.DataFrame(records))
        for row in datasets.itertuples():
            self.assertAlmostEqual(row.sum_oof_log_score, row.N * row.mean_oof_log_score)
        for row in aggregate.itertuples():
            self.assertAlmostEqual(row.sum_oof_log_score, math.log(.5) + 3 * math.log(.25))
            self.assertAlmostEqual(row.balanced_mean_oof_log_score, (math.log(.5) + math.log(.25)) / 2)


if __name__ == "__main__":
    unittest.main()
