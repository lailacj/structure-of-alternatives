"""Force all table trigger/query candidates into cluster scoring vocabularies."""

from __future__ import annotations

import argparse
from pathlib import Path


def _read(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_augmented(base: Path, required: set[str], output: Path) -> None:
    values = _read(base)
    seen = set(values)
    values.extend(sorted(required.difference(seen)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(values) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-unigram-vocab", type=Path, required=True)
    parser.add_argument("--base-bigram-vocab", type=Path, required=True)
    parser.add_argument("--required-candidates", type=Path, required=True)
    parser.add_argument("--output-unigram-vocab", type=Path, required=True)
    parser.add_argument("--output-bigram-vocab", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = _read(args.required_candidates)
    invalid = [value for value in required if len(value.split()) not in {1, 2}]
    if invalid:
        raise ValueError(f"Only unigram/bigram candidates are supported: {invalid}")
    _write_augmented(args.base_unigram_vocab, {value for value in required if len(value.split()) == 1}, args.output_unigram_vocab)
    _write_augmented(args.base_bigram_vocab, {value for value in required if len(value.split()) == 2}, args.output_bigram_vocab)
    print(f"[complete] required_candidates={len(required)}")


if __name__ == "__main__":
    main()
