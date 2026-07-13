"""Regression checks for the committed cross-dataset canonical tables."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PIPELINE_DIR / "code"
CANONICAL_DIR = PIPELINE_DIR / "canonical_data"
sys.path.insert(0, str(CODE_DIR))

from canonical_observations import validate_canonical_observations  # noqa: E402


REVISION = "453ed1575b739b5b03ce3758b23befdb0967f40e"


class CrossDatasetCanonicalOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.focus = pd.read_csv(CANONICAL_DIR / "novel_focus_observations.csv")
        cls.hu = pd.read_csv(CANONICAL_DIR / "hu_2023_observations.csv")
        cls.rnx = pd.read_csv(CANONICAL_DIR / "ronai_xiang_2024_observations.csv")
        cls.all = pd.read_csv(CANONICAL_DIR / "all_observations.csv")

    def test_all_tables_validate_with_complete_model_provenance(self) -> None:
        for table in [self.focus, self.hu, self.rnx, self.all]:
            validate_canonical_observations(table, require_complete_provenance=True)
            self.assertEqual(table["model_revision"].unique().tolist(), [REVISION])

    def test_expected_dataset_and_condition_grains(self) -> None:
        self.assertEqual(len(self.focus), 480)
        self.assertEqual(len(self.hu), 309)
        self.assertEqual(len(self.rnx), 300)
        self.assertEqual(len(self.all), 1089)
        self.assertEqual(
            self.rnx.groupby("condition").size().to_dict(),
            {"ESI": 60, "Eonly": 60, "Eonlystrong": 60, "Estrong": 60, "Eweak": 60},
        )
        self.assertEqual(
            self.hu.groupby("dataset").size().to_dict(),
            {"g18": 70, "pvt21": 50, "rx22": 60, "vt16": 129},
        )

    def test_frame_applicability_matches_frozen_definition(self) -> None:
        self.assertTrue(self.focus["x_but_not_y_applicable"].all())
        self.assertTrue(self.hu["x_but_not_y_applicable"].all())
        self.assertFalse(self.rnx["x_but_not_y_applicable"].any())
        self.assertTrue(self.rnx["x_but_not_y_logprob_sum"].isna().all())

    def test_hu_count_availability_is_not_fabricated(self) -> None:
        status = self.hu.groupby(["dataset", "human_count_status"]).size().to_dict()
        self.assertEqual(
            status,
            {
                ("g18", "rate_only"): 70,
                ("pvt21", "rate_only"): 50,
                ("rx22", "exact"): 60,
                ("vt16", "rate_only"): 129,
            },
        )
        rate_only = self.hu["human_count_status"].eq("rate_only")
        self.assertTrue(self.hu.loc[rate_only, ["human_yes", "human_total"]].isna().all(axis=None))

    def test_hu_original_analysis_filter_matches_source_notebook(self) -> None:
        included = self.hu.loc[self.hu["hu_original_analysis_included"]]
        self.assertEqual(
            included.groupby("dataset").size().to_dict(),
            {"g18": 67, "pvt21": 50, "rx22": 57, "vt16": 117},
        )
        self.assertEqual(
            included.groupby("dataset")["scale_id"].nunique().to_dict(),
            {"g18": 67, "pvt21": 50, "rx22": 57, "vt16": 39},
        )
        self.assertEqual(
            set(self.hu["analysis_inclusion_status"].unique()),
            {
                "included_hu_original_string_analysis",
                "excluded_hu_original_missing_gpt2_surprisal",
                "excluded_hu_original_source_dropna_missing_lsa",
            },
        )
        source_dropna = self.hu.loc[
            self.hu["analysis_inclusion_status"].eq(
                "excluded_hu_original_source_dropna_missing_lsa"
            )
        ]
        self.assertEqual(source_dropna["scale_id"].unique().tolist(), ["unsettling/horrific"])
        self.assertEqual(len(source_dropna), 3)

    def test_hu_frame_specific_inflections_are_explicit(self) -> None:
        query_differs = self.hu["query"].str.lower().ne(
            self.hu["x_but_not_y_query"].str.lower()
        )
        self.assertEqual(int(query_differs.sum()), 33)


if __name__ == "__main__":
    unittest.main()
