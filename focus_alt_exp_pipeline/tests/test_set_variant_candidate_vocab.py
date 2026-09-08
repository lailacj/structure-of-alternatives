"""Tests for bounded set-variant candidate-vocabulary augmentation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from build_set_variant_candidate_vocab import build_candidate_vocabs  # noqa: E402


class SetVariantCandidateVocabTests(unittest.TestCase):
    def test_force_includes_missing_candidates_without_reordering_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = root / "required.txt"
            unigrams = root / "unigrams.txt"
            bigrams = root / "bigrams.txt"
            required.write_text("apple\nnew word\nrare\n", encoding="utf-8")
            unigrams.write_text("apple\npear\n", encoding="utf-8")
            bigrams.write_text("old word\n", encoding="utf-8")
            out_uni, out_bi, manifest = build_candidate_vocabs(
                required_path=required, unigram_path=unigrams, bigram_path=bigrams
            )
            self.assertEqual(out_uni, ["apple", "pear", "rare"])
            self.assertEqual(out_bi, ["old word", "new word"])
            self.assertEqual(manifest["counts"]["added_unigrams"], 1)
            self.assertEqual(manifest["counts"]["added_bigrams"], 1)
            self.assertEqual(manifest["coverage"]["missing_unigrams"], [])
            self.assertEqual(manifest["coverage"]["missing_bigrams"], [])

    def test_rejects_candidates_longer_than_bigrams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = root / "required.txt"
            unigrams = root / "unigrams.txt"
            bigrams = root / "bigrams.txt"
            required.write_text("three word candidate\n", encoding="utf-8")
            unigrams.write_text("apple\n", encoding="utf-8")
            bigrams.write_text("old word\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_candidate_vocabs(
                    required_path=required, unigram_path=unigrams, bigram_path=bigrams
                )


if __name__ == "__main__":
    unittest.main()
