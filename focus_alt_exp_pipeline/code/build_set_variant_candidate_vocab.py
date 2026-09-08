"""Build bounded Qwen candidate vocabularies for the set-variant analysis.

The output preserves the supplied unigram vocabulary and bounded global
bigram support, then force-includes every normalized trigger/query candidate
listed by ``build_set_variant_scoring_manifest.py``.  It never reads the raw
full bigram vocabulary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PIPELINE_DIR.parent
DEFAULT_REQUIRED = PIPELINE_DIR / "scoring_manifests" / "set_variant_qwen" / "required_candidates.txt"
DEFAULT_UNIGRAM = ROOT_DIR.parent / "ngrams" / "google_ngram_frequency_info" / "vocab_1gram.txt"
DEFAULT_BIGRAM = ROOT_DIR.parent / "ngrams" / "qwen_bigram_support" / "vocab_2gram.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR.parent / "ngrams" / "set_variant_qwen_support"


def _load_vocab(path: Path, *, expected_words: int | None = None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        tokens = [line.rstrip("\n") for line in stream]
    if any(not token.strip() for token in tokens):
        raise ValueError(f"Vocabulary contains empty lines: {path}")
    normalized = [token.strip().lower() for token in tokens]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Vocabulary contains duplicate normalized tokens: {path}")
    if expected_words is not None:
        wrong = [token for token in normalized if len(token.split()) != expected_words]
        if wrong:
            raise ValueError(f"Expected {expected_words}-word tokens in {path}; examples: {wrong[:5]}")
    return normalized


def build_candidate_vocabs(
    *, required_path: Path, unigram_path: Path, bigram_path: Path
) -> tuple[list[str], list[str], dict]:
    required = _load_vocab(required_path)
    too_long = sorted(token for token in required if len(token.split()) > 2)
    if too_long:
        raise ValueError(f"Only unigram/bigram candidates are supported: {too_long}")

    unigrams = _load_vocab(unigram_path, expected_words=1)
    bigrams = _load_vocab(bigram_path, expected_words=2)
    required_unigrams = sorted(token for token in required if len(token.split()) == 1)
    required_bigrams = sorted(token for token in required if len(token.split()) == 2)
    unigram_set, bigram_set = set(unigrams), set(bigrams)
    added_unigrams = sorted(set(required_unigrams).difference(unigram_set))
    added_bigrams = sorted(set(required_bigrams).difference(bigram_set))
    augmented_unigrams = [*unigrams, *added_unigrams]
    augmented_bigrams = [*bigrams, *added_bigrams]
    manifest = {
        "schema_version": "focus-alternatives-set-variant-candidate-vocab/v1",
        "inputs": {
            "required_candidates_path": str(required_path.resolve()),
            "base_unigram_path": str(unigram_path.resolve()),
            "bounded_global_bigram_path": str(bigram_path.resolve()),
        },
        "counts": {
            "required_candidates": len(required),
            "required_unigrams": len(required_unigrams),
            "required_bigrams": len(required_bigrams),
            "base_unigrams": len(unigrams),
            "base_bigrams": len(bigrams),
            "added_unigrams": len(added_unigrams),
            "added_bigrams": len(added_bigrams),
            "final_unigrams": len(augmented_unigrams),
            "final_bigrams": len(augmented_bigrams),
            "final_total": len(augmented_unigrams) + len(augmented_bigrams),
        },
        "added_candidates": {"unigrams": added_unigrams, "bigrams": added_bigrams},
        "coverage": {
            "missing_unigrams": sorted(set(required_unigrams).difference(augmented_unigrams)),
            "missing_bigrams": sorted(set(required_bigrams).difference(augmented_bigrams)),
        },
    }
    return augmented_unigrams, augmented_bigrams, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-candidates", type=Path, default=DEFAULT_REQUIRED)
    parser.add_argument("--base-unigram-vocab", type=Path, default=DEFAULT_UNIGRAM)
    parser.add_argument("--base-bigram-vocab", type=Path, default=DEFAULT_BIGRAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unigrams, bigrams, manifest = build_candidate_vocabs(
        required_path=args.required_candidates,
        unigram_path=args.base_unigram_vocab,
        bigram_path=args.base_bigram_vocab,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_unigram = args.output_dir / "vocab_1gram.txt"
    output_bigram = args.output_dir / "vocab_2gram.txt"
    manifest["outputs"] = {
        "unigram_vocab_path": str(output_unigram.resolve()),
        "bigram_vocab_path": str(output_bigram.resolve()),
        "manifest_path": str((args.output_dir / "candidate_vocab_manifest.json").resolve()),
    }
    output_unigram.write_text("\n".join(unigrams) + "\n", encoding="utf-8")
    output_bigram.write_text("\n".join(bigrams) + "\n", encoding="utf-8")
    (args.output_dir / "candidate_vocab_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[complete] unigrams={len(unigrams)} bigrams={len(bigrams)} "
        f"total={len(unigrams) + len(bigrams)} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
