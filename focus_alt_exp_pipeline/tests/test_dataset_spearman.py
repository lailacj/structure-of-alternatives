"""Validate grouping, aggregation, and scientific parity for non-focus Spearman."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / 'code'))
from build_linking_structure_tables import build_dataset_spearman, direct_analysis_units
from evaluate_set_variant_grid import summarize_correlations


class DatasetSpearmanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = pd.read_csv(PIPELINE / 'scoring_manifests/set_variant_qwen/source_rows.csv')
        cls.oof = pd.read_csv(PIPELINE / 'results/set_variant_qwen/cv_results/oof_predictions.csv')
        cls.wide, cls.summary, cls.paired = build_dataset_spearman(cls.source, cls.oof)

    def test_group_coverage_and_independent_rank_parity(self):
        expected = {'hu_vt16':39,'hu_g18':67,'hu_pvt21':50,'hu_rx22':57,
                    **{f'rnx_{c}':60 for c in ['esi','eweak','estrong','eonly','eonlystrong']}}
        self.assertEqual(len(self.wide),9)
        self.assertEqual(len(self.summary),81)
        for row in self.summary.itertuples():
            if row.status == 'not_applicable':
                self.assertTrue(row.analysis_dataset_id.startswith('rnx_'))
                self.assertEqual(row.model,'X but not Y')
                self.assertEqual(row.n,0)
                continue
            self.assertEqual(row.n,expected[row.analysis_dataset_id])
            pair=self.paired.loc[self.paired.analysis_dataset_id.eq(row.analysis_dataset_id)&self.paired.model.eq(row.model)]
            self.assertEqual(len(pair),row.n)
            self.assertEqual(pair.analysis_unit_id.nunique(),row.n)
            # Independent rank construction by counting greater and equal values.
            def reference_ranks(values):
                v=np.asarray(values)
                return np.array([(v > x).sum() + ((v == x).sum()-1)/2 for x in v])
            expected_rho=np.corrcoef(reference_ranks(pair.human_rate),reference_ranks(pair.model_probability))[0,1]
            self.assertAlmostEqual(row.spearman_rho,expected_rho)
            self.assertAlmostEqual(pair.human_rank.mean(),(row.n-1)/2)
            self.assertAlmostEqual(pair.model_rank.mean(),(row.n-1)/2)
        self.assertNotIn('novel_focus',set(self.paired.analysis_dataset_id))

    def test_van_tiel_averages_probabilities_before_ranking(self):
        direct=direct_analysis_units(self.source).query("analysis_dataset_id == 'hu_vt16'")
        pairs=self.paired.query("analysis_dataset_id == 'hu_vt16' and model == 'No linking structure'")
        joined=direct.merge(pairs,on='analysis_unit_id',validate='one_to_one')
        np.testing.assert_allclose(joined['No linking structure'],joined.model_probability)
        # Every included scale is represented by three source templates, but only one rank.
        raw=self.source.query("dataset == 'vt16' and hu_original_analysis_included == True")
        self.assertEqual(len(raw),117)
        self.assertEqual(len(pairs),39)

    def test_shuffling_and_other_conditions_do_not_change_group(self):
        _,summary,_=build_dataset_spearman(self.source.sample(frac=1,random_state=1),self.oof.sample(frac=1,random_state=2))
        pd.testing.assert_frame_equal(self.summary,summary)
        changed=self.oof.copy()
        changed.loc[changed.analysis_dataset_id.eq('rnx_eweak'),'set_probability']=0
        _,summary,_=build_dataset_spearman(self.source,changed)
        original=self.summary.query("analysis_dataset_id != 'rnx_eweak'")
        pd.testing.assert_frame_equal(original,summary.query("analysis_dataset_id != 'rnx_eweak'"))
        constant=summary.query("analysis_dataset_id == 'rnx_eweak' and model == 'Set Top-K'").iloc[0]
        self.assertTrue(pd.isna(constant.spearman_rho))
        self.assertEqual(constant.status,'constant_human_or_model')

    def test_missing_duplicate_and_nonfinite_predictions_fail(self):
        idx=self.oof.index[self.oof.analysis_dataset_id.eq('hu_g18')][0]
        bad=self.oof.copy();bad.loc[idx,'set_probability']=np.inf
        for frame in (bad,self.oof.drop(idx),pd.concat([self.oof,self.oof.loc[[idx]]])):
            with self.assertRaises(ValueError):
                build_dataset_spearman(self.source,frame)

    def test_sampled_evaluator_matches_linking_table(self):
        summary=summarize_correlations(self.oof)
        for row in summary.itertuples():
            if row.analysis_dataset_id=='novel_focus':
                self.assertTrue(pd.isna(row.set_spearman_rho))
                self.assertEqual(row.spearman_grouping,'not_evaluated')
                continue
            label='Set Top-K' if row.variant=='top_k' else 'Set Top-p'
            expected=self.summary.loc[self.summary.analysis_dataset_id.eq(row.analysis_dataset_id)&self.summary.model.eq(label),'spearman_rho'].iloc[0]
            self.assertAlmostEqual(row.set_spearman_rho,expected)

if __name__ == '__main__':
    unittest.main()
