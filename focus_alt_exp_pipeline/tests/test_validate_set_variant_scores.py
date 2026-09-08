"""Tests for strict set-variant Qwen score validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from validate_set_variant_qwen_scores import validate_scores  # noqa: E402


class SetVariantScoreValidationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        manifest_dir, scores = root / "manifest", root / "scores"
        manifest_dir.mkdir()
        scores.mkdir()
        pd.DataFrame([{"prompt_id": "prompt_a", "generation_prompt": "A"}]).to_csv(
            manifest_dir / "prompts.csv", index=False
        )
        pd.DataFrame([{"prompt_id": "prompt_a", "trigger": "apple", "query": "rare"}]).to_csv(
            manifest_dir / "source_rows.csv", index=False
        )
        unigram, bigram = root / "unigram.txt", root / "bigram.txt"
        unigram.write_text("apple\nrare\n", encoding="utf-8")
        bigram.write_text("old word\n", encoding="utf-8")
        (scores / "vocab_manifest.json").write_text(json.dumps({
            "dtype": "float32",
            "sources": [
                {"name": "1gram", "path": str(unigram), "count": 2, "offset": 0},
                {"name": "2gram", "path": str(bigram), "count": 1, "offset": 2},
            ],
            "total_count": 3,
        }), encoding="utf-8")
        np.save(scores / "prompt_a.log_probs.npy", np.array([-1.0, -2.0, -3.0], dtype=np.float32))
        (scores / "prompt_a.progress.json").write_text(json.dumps({
            "sources": {"1gram": {"done": True}, "2gram": {"done": True}}
        }), encoding="utf-8")
        (scores / "prompt_a.meta.json").write_text(json.dumps({
            "context": "prompt_a", "target_vocab_size": 3
        }), encoding="utf-8")
        return manifest_dir, scores

    def test_accepts_complete_finite_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir, scores = self._fixture(Path(directory))
            report = validate_scores(
                manifest_dir=manifest_dir, log_probs_dir=scores, allow_incomplete=False
            )
            self.assertTrue(report["ready"])
            self.assertEqual(report["complete_prompts"], 1)

    def test_reports_nonfinite_required_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir, scores = self._fixture(Path(directory))
            np.save(scores / "prompt_a.log_probs.npy", np.array([-1.0, np.nan, -3.0], dtype=np.float32))
            report = validate_scores(
                manifest_dir=manifest_dir, log_probs_dir=scores, allow_incomplete=True
            )
            issues = {problem["issue"] for problem in report["problems"]}
            self.assertIn("nonfinite_scores", issues)
            self.assertIn("nonfinite_required_candidates", issues)
            self.assertFalse(report["ready"])


if __name__ == "__main__":
    unittest.main()
