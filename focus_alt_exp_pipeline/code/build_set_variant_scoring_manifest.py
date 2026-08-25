"""Build cluster inputs for the sampled-prefix Top-K / Top-p analysis.

The output deliberately contains prompts only; it contains no human outcomes.
Use it with ``precompute_qwen_vocab_log_probs.py`` on the cluster to score one
candidate-word distribution per unique neutral-frame prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_DIR / "canonical_data" / "all_observations.csv"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "scoring_manifests" / "set_variant_qwen"


def _source_row_key(row: pd.Series) -> str:
    fields = [
        row["dataset_family"], row["dataset"], row["condition"],
        row["item_id"], row["context_id"], row["generation_frame"],
    ]
    return json.dumps([str(value) for value in fields], separators=(",", ":"))


def _prompt_id(prompt: str) -> str:
    return "prompt_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]


def build_manifest(canonical: pd.DataFrame, *, bigram_vocab: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"generation_prompt", "trigger", "query", "dataset_family", "dataset", "condition", "item_id", "context_id", "generation_frame"}
    missing = required.difference(canonical.columns)
    if missing:
        raise ValueError(f"Canonical observations are missing columns: {sorted(missing)}")
    if canonical["generation_prompt"].isna().any():
        raise ValueError("Canonical observations contain empty generation prompts")

    source_rows = canonical.copy()
    source_rows["generation_prompt"] = source_rows["generation_prompt"].astype(str)
    source_rows["source_row_key"] = source_rows.apply(_source_row_key, axis=1)
    if source_rows["source_row_key"].duplicated().any():
        raise ValueError("Canonical source-row keys are not unique")
    source_rows["prompt_id"] = source_rows["generation_prompt"].map(_prompt_id)

    too_long = []
    required_bigrams: dict[str, set[str]] = {}
    for _, row in source_rows.iterrows():
        prompt_id = str(row["prompt_id"])
        for candidate in (str(row["trigger"]), str(row["query"])):
            token = candidate.strip().lower()
            words = token.split()
            if len(words) > 2:
                too_long.append(token)
            if len(words) == 2:
                required_bigrams.setdefault(prompt_id, set()).add(token)
    if too_long:
        raise ValueError(
            "The current scorer supports unigram/bigram candidate vocabularies only; "
            f"found longer candidates: {sorted(set(too_long))}"
        )

    prompts = source_rows[["prompt_id", "generation_prompt"]].drop_duplicates().sort_values("prompt_id")
    manifest = {
        "schema_version": "focus-alternatives-set-variant-support/v1",
        "outputs": {"global_vocab_path": str(bigram_vocab)},
        "contexts": {
            str(row.prompt_id): {"required_bigrams": sorted(required_bigrams.get(str(row.prompt_id), set()))}
            for row in prompts.itertuples(index=False)
        },
    }
    return source_rows, prompts, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bigram-vocab", type=Path, required=True,
        help="Cluster path to the global candidate bigram vocabulary to score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = pd.read_csv(args.input)
    source_rows, prompts, manifest = build_manifest(canonical, bigram_vocab=args.bigram_vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows.to_csv(args.output_dir / "source_rows.csv", index=False)
    prompts.to_csv(args.output_dir / "prompts.csv", index=False)
    with (args.output_dir / "selection_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    required = sorted({str(value).strip().lower() for value in pd.concat([source_rows["trigger"], source_rows["query"]])})
    (args.output_dir / "required_candidates.txt").write_text("\n".join(required) + "\n", encoding="utf-8")
    print(f"[complete] prompts={len(prompts)} source_rows={len(source_rows)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
