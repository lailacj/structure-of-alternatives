"""Scientific invariants and real-data coverage for context Spearman."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "code"))
from evaluate_focus_spearman import evaluate, rank_correlation, HUMAN_FILE


class SpearmanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.human = pd.read_csv(HUMAN_FILE)
        cls.source = pd.read_csv(PIPELINE / "scoring_manifests/set_variant_qwen/source_rows.csv")
        cls.oof = pd.read_csv(PIPELINE / "results/set_variant_qwen/cv_results/oof_predictions.csv")
        cls.tables = evaluate(cls.human, cls.source, cls.oof)

    def test_known_rankings(self):
        self.assertAlmostEqual(rank_correlation(range(6), range(6))[0], 1)
        self.assertAlmostEqual(rank_correlation(range(6), range(5, -1, -1))[0], -1)
        self.assertAlmostEqual(rank_correlation(range(6), [0, 2, 1, 3, 4, 5])[0], 33/35)
        self.assertAlmostEqual(rank_correlation([1, 1, 3], [1, 2, 3])[0], np.sqrt(3)/2)
        self.assertTrue(np.isnan(rank_correlation([1, 1], [2, 3])[0]))
        with self.assertRaises(ValueError):
            rank_correlation([1, np.nan], [2, 3])

    def test_coverage_and_fridge(self):
        self.assertEqual(len(self.tables['word_spearman_by_context']), 16)
        self.assertTrue(self.tables['word_spearman_by_context'].n.eq(6).all())
        self.assertTrue(self.tables['negation_spearman_by_context'].n.eq(30).all())
        fridge = self.tables['word_paired_ranks'].query("context == 'fridge'").sort_values('human_rank')
        self.assertEqual(fridge.word.tolist(), ['water', 'juice', 'milk', 'yogurt', 'ketchup', 'meat'])
        for _, row in self.tables['mean_within_context_spearman'].iterrows():
            table = self.tables['word_spearman_by_context'] if row.measure == 'word_ranking' else self.tables['negation_spearman_by_context'].loc[lambda x: x.structure.eq(row.structure) & x.variant.eq(row.variant)]
            self.assertAlmostEqual(row.mean_within_context_spearman, table.spearman_rho.mean())
            self.assertEqual(row.valid_contexts, table.spearman_rho.count())

    def test_order_invariance(self):
        shuffled = evaluate(self.human.sample(frac=1, random_state=1), self.source.sample(frac=1, random_state=2), self.oof.sample(frac=1, random_state=3))
        for name in ['word_spearman_by_context', 'negation_spearman_by_context', 'mean_within_context_spearman']:
            pd.testing.assert_frame_equal(self.tables[name], shuffled[name])

    def test_missing_scores_and_pairs_fail(self):
        for bad_source in [self.source.iloc[1:], self.source.assign(query_logprob_sum=np.nan)]:
            with self.assertRaises(ValueError):
                evaluate(self.human, bad_source, self.oof)
        missing = self.oof.drop(self.oof.index[self.oof.analysis_dataset_id.eq('novel_focus')][0])
        with self.assertRaises(ValueError):
            evaluate(self.human, self.source, missing)

    def test_conflicting_human_ranks_fail(self):
        bad = self.human.copy()
        bad.loc[0, 'trigger_relevance'] = 5
        with self.assertRaises(ValueError):
            evaluate(bad, self.source, self.oof)

    def test_within_context_not_pooled(self):
        # Perfect rankings in each group even though between-group order is reversed.
        self.assertAlmostEqual(rank_correlation([.9,.8,.7], [.3,.2,.1])[0], 1)
        self.assertAlmostEqual(rank_correlation([.3,.2,.1], [.9,.8,.7])[0], 1)
        self.assertLess(rank_correlation([.9,.8,.7,.3,.2,.1], [.3,.2,.1,.9,.8,.7])[0], 0)

if __name__ == '__main__':
    unittest.main()
