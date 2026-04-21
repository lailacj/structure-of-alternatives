"""Build filtered cloze-vocab resources for the constrained follow-up run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from .data_utils import (
        clean_word,
        extract_support_tokens_by_context,
        normalize_unique_tokens,
        prepare_experimental_data,
        resolve_context_col,
    )
except ImportError:
    from data_utils import (
        clean_word,
        extract_support_tokens_by_context,
        normalize_unique_tokens,
        prepare_experimental_data,
        resolve_context_col,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CLOZE_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "cloze_data" / "all_cloze_prob_data_preprocessed.csv"
DEFAULT_EXPERIMENTAL_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_FILTERED_CLOZE_OUTPUT = (
    ROOT_DIR / "focus_alt_exp_pipeline" / "cloze_data" / "all_cloze_prob_data_preprocessed_unigram_bigram.csv"
)
DEFAULT_FILTERED_EXPERIMENTAL_OUTPUT = (
    ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe_supported_by_cloze_unigram_bigram.csv"
)
DEFAULT_QWEN_SUPPORT_DIR = ROOT_DIR.parent / "ngrams" / "qwen_cloze_vocab_support"


def _safe_stem(text: str) -> str:
    keep = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _normalize_phrase(value: object) -> str:
    token = clean_word(str(value))
    return " ".join(token.split())


def _word_count(value: str) -> int:
    token = str(value).strip()
    if not token:
        return 0
    return len(token.split())


def _build_filtered_cloze_dataframe(
    cloze_df: pd.DataFrame,
    *,
    context_col: str,
    word_col: str,
    prob_col: str,
    max_words: int,
) -> pd.DataFrame:
    required = {context_col, word_col, prob_col}
    missing = required.difference(cloze_df.columns)
    if missing:
        raise ValueError(f"Missing columns in cloze data: {sorted(missing)}")

    prepared = cloze_df[[context_col, word_col, prob_col]].copy()
    prepared[context_col] = prepared[context_col].astype(str).str.strip()
    prepared[word_col] = prepared[word_col].apply(_normalize_phrase)
    prepared[prob_col] = pd.to_numeric(prepared[prob_col], errors="raise")
    prepared["n_words"] = prepared[word_col].apply(_word_count)
    prepared = prepared[
        (prepared[word_col] != "") & (prepared["n_words"] > 0) & (prepared["n_words"] <= max_words)
    ].copy()

    aggregated = (
        prepared.groupby([context_col, word_col], as_index=False, sort=False)
        .agg(
            original_cloze_probability=(prob_col, "sum"),
            source_rows=(prob_col, "size"),
        )
    )
    aggregated["n_words"] = aggregated[word_col].apply(_word_count)
    aggregated["cloze_probability"] = aggregated.groupby(context_col, sort=False)[
        "original_cloze_probability"
    ].transform(lambda values: values / values.sum())
    return aggregated.sort_values(
        [context_col, "cloze_probability", word_col],
        ascending=[True, False, True],
        ignore_index=True,
    )[
        [
            context_col,
            word_col,
            "cloze_probability",
            "original_cloze_probability",
            "source_rows",
            "n_words",
        ]
    ]


def _filter_experimental_dataframe(
    experimental_df: pd.DataFrame,
    *,
    support_by_context: Dict[str, List[str]],
    filter_mode: str,
) -> pd.DataFrame:
    prepared = prepare_experimental_data(experimental_df)
    context_col = resolve_context_col(prepared)
    query_col = "cleaned_query" if "cleaned_query" in prepared.columns else "query"
    trigger_col = "cleaned_trigger" if "cleaned_trigger" in prepared.columns else "trigger"
    support_sets = {context: set(tokens) for context, tokens in support_by_context.items()}

    keep_mask = []
    for _, row in prepared.iterrows():
        context = str(row[context_col])
        query = str(row[query_col])
        trigger = str(row[trigger_col])
        context_support = support_sets.get(context, set())
        keep = query in context_support
        if filter_mode == "query_and_trigger":
            keep = keep and (trigger in context_support)
        keep_mask.append(keep)

    return prepared.loc[keep_mask].copy()


def _write_lines(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def _build_qwen_support_resources(
    filtered_cloze_df: pd.DataFrame,
    *,
    context_col: str,
    word_col: str,
    output_dir: Path,
    source_cloze_path: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir = output_dir / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)

    support_by_context = extract_support_tokens_by_context(
        filtered_cloze_df,
        context_col=context_col,
        word_col=word_col,
    )
    global_vocab = normalize_unique_tokens(
        [token for tokens in support_by_context.values() for token in tokens]
    )
    unigram_vocab = sorted(token for token in global_vocab if _word_count(token) == 1)
    bigram_vocab = sorted(token for token in global_vocab if _word_count(token) == 2)

    vocab_1gram_path = output_dir / "vocab_1gram.txt"
    vocab_2gram_path = output_dir / "vocab_2gram.txt"
    manifest_path = output_dir / "selection_manifest.json"

    _write_lines(vocab_1gram_path, unigram_vocab)
    _write_lines(vocab_2gram_path, bigram_vocab)

    context_payload = {}
    for context, tokens in support_by_context.items():
        required_bigrams = [token for token in tokens if _word_count(token) == 2]
        context_bigram_path = contexts_dir / f"{_safe_stem(context)}.selected_bigrams.tsv"
        pd.DataFrame({"token": required_bigrams}).to_csv(
            context_bigram_path,
            sep="\t",
            index=False,
        )
        context_payload[context] = {
            "required_tokens": tokens,
            "required_bigrams": required_bigrams,
            "family_first_words": sorted({token.split(" ", 1)[0] for token in tokens}),
            "selected_bigrams_path": str(context_bigram_path),
            "selected_bigram_count": len(required_bigrams),
            "family_bigram_count": len(required_bigrams),
            "background_bigram_count": 0,
            "unsupported_tokens": [],
        }

    manifest = {
        "version": "cloze_vocab_restricted_v1",
        "source_cloze_path": str(source_cloze_path),
        "selection_strategy": "cloze_unigram_bigram_support",
        "outputs": {
            "unigram_vocab_path": str(vocab_1gram_path),
            "global_vocab_path": str(vocab_2gram_path),
            "global_counts_path": None,
            "background_pool_path": None,
            "contexts_dir": str(contexts_dir),
            "selection_manifest_path": str(manifest_path),
        },
        "contexts": context_payload,
        "global_unigram_vocab_size": len(unigram_vocab),
        "global_bigram_vocab_size": len(bigram_vocab),
        "global_vocab_size": len(global_vocab),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered cloze-vocabulary resources for the constrained follow-up run"
    )
    parser.add_argument("--cloze-data", type=Path, default=DEFAULT_CLOZE_DATA)
    parser.add_argument("--experimental-data", type=Path, default=DEFAULT_EXPERIMENTAL_DATA)
    parser.add_argument("--cloze-context-col", type=str, default="context")
    parser.add_argument("--cloze-word-col", type=str, default="word")
    parser.add_argument("--cloze-prob-col", type=str, default="cloze_probability")
    parser.add_argument("--max-words", type=int, default=2)
    parser.add_argument(
        "--experimental-filter-mode",
        choices=["query", "query_and_trigger"],
        default="query_and_trigger",
        help="How to filter human trials against the filtered cloze support.",
    )
    parser.add_argument("--filtered-cloze-output", type=Path, default=DEFAULT_FILTERED_CLOZE_OUTPUT)
    parser.add_argument(
        "--filtered-experimental-output",
        type=Path,
        default=DEFAULT_FILTERED_EXPERIMENTAL_OUTPUT,
    )
    parser.add_argument("--qwen-support-dir", type=Path, default=DEFAULT_QWEN_SUPPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_words <= 0:
        raise ValueError("--max-words must be > 0")

    cloze_df = pd.read_csv(args.cloze_data)
    experimental_df = pd.read_csv(args.experimental_data)

    filtered_cloze_df = _build_filtered_cloze_dataframe(
        cloze_df,
        context_col=args.cloze_context_col,
        word_col=args.cloze_word_col,
        prob_col=args.cloze_prob_col,
        max_words=args.max_words,
    )
    support_by_context = extract_support_tokens_by_context(
        filtered_cloze_df,
        context_col=args.cloze_context_col,
        word_col=args.cloze_word_col,
    )
    filtered_experimental_df = _filter_experimental_dataframe(
        experimental_df,
        support_by_context=support_by_context,
        filter_mode=args.experimental_filter_mode,
    )
    manifest = _build_qwen_support_resources(
        filtered_cloze_df,
        context_col=args.cloze_context_col,
        word_col=args.cloze_word_col,
        output_dir=args.qwen_support_dir,
        source_cloze_path=args.filtered_cloze_output,
    )

    args.filtered_cloze_output.parent.mkdir(parents=True, exist_ok=True)
    filtered_cloze_df.to_csv(args.filtered_cloze_output, index=False)

    args.filtered_experimental_output.parent.mkdir(parents=True, exist_ok=True)
    filtered_experimental_df.to_csv(args.filtered_experimental_output, index=False)

    print("Built cloze-vocab-restricted resources.")
    print(f"filtered_cloze_output={args.filtered_cloze_output}")
    print(f"filtered_experimental_output={args.filtered_experimental_output}")
    print(f"qwen_support_dir={args.qwen_support_dir}")
    print(f"filtered_cloze_rows={len(filtered_cloze_df)}")
    print(f"filtered_cloze_contexts={filtered_cloze_df[args.cloze_context_col].nunique()}")
    print(f"filtered_global_vocab_size={manifest['global_vocab_size']}")
    print(f"filtered_global_unigrams={manifest['global_unigram_vocab_size']}")
    print(f"filtered_global_bigrams={manifest['global_bigram_vocab_size']}")
    filtered_context_col = resolve_context_col(filtered_experimental_df)
    print(f"filtered_experimental_rows={len(filtered_experimental_df)}")
    print(f"filtered_experimental_contexts={filtered_experimental_df[filtered_context_col].nunique()}")


if __name__ == "__main__":
    main()
