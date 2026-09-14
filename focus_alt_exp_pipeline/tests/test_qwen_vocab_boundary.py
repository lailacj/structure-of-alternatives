"""Regression checks for the full-vocabulary prompt/continuation boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import precompute_qwen_vocab_log_probs as scorer
from score_qwen_scoring_manifest import split_prompt_boundary, continuation_token_ids


class Tokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(text.encode()))


class BoundaryTests(unittest.TestCase):
    def test_single_space_for_every_candidate_shape(self):
        tokenizer = Tokenizer()
        for prompt in ["Sure, I have my ", "Sure, I have my", "Sure, I have my  \n"]:
            for candidate in ["mask", "pretzel stand", " blankets ", "electronics store"]:
                prefix = scorer._prompt_prefix(prompt)
                ids = scorer._vocab_continuation_ids(tokenizer, prefix, candidate)
                assembled = bytes(tokenizer(prefix).input_ids + ids).decode()
                self.assertEqual(assembled, "Sure, I have my " + candidate.strip())
                reference_prefix, continuation = split_prompt_boundary(prefix + " ", candidate)
                expected, _ = continuation_token_ids(tokenizer, prefix=reference_prefix, continuation=continuation)
                self.assertEqual(ids, expected)

    def test_full_text_tokenization_used_instead_of_separate_candidate_encoding(self):
        class ContextTokenizer:
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids={"I have": [1, 2], "I have blankets": [1, 2, 3, 4], " blankets": [99]}[text])
        self.assertEqual(scorer._vocab_continuation_ids(ContextTokenizer(), "I have", "blankets"), [3, 4])

    def test_boundary_merge_and_empty_text_rejected(self):
        class MergingTokenizer:
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=[1] if text == "prefix" else [2, 3])
        with self.assertRaisesRegex(ValueError, "merged across"):
            scorer._vocab_continuation_ids(MergingTokenizer(), "prefix", "candidate")
        with self.assertRaises(ValueError):
            scorer._prompt_prefix(" \n")
        with self.assertRaises(ValueError):
            scorer._vocab_continuation_ids(Tokenizer(), "prefix", " ")

    def test_legacy_arrays_cannot_be_resumed_or_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ["scores.npy", "progress.json", "meta.json"]]
            scorer._check_scoring_boundary(*paths)
            for path in paths:
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError, "legacy"):
                    scorer._check_scoring_boundary(*paths)
                self.assertEqual(path.read_text(), '{}')
            paths[2].write_text(json.dumps({"scoring_boundary_version": scorer.SCORING_BOUNDARY_VERSION}))
            scorer._check_scoring_boundary(*paths)


if __name__ == "__main__":
    unittest.main()
