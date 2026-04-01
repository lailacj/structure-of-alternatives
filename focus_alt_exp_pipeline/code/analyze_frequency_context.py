"""Deep-dive diagnostics for the frequency model within a context.

This script focuses on a single context (for example, ``bag``) and reports:

- the n-gram count used by the frequency model for each context token
- each token's share among all supported frequency-model tokens
- each token's share within the selected context
- for every ordered query/trigger pair, the exact probability that the query
  ranks above the trigger under the frequency model
- the empirical ordering probability observed in
  ``ordering_results_frequency.csv``

The exact pairwise probability comes from the Plackett-Luce / weighted without
replacement sampler used by the global ``FrequencySampler`` baseline.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import pandas as pd

try:
    from .data_utils import (
        clean_word,
        normalize_unique_tokens,
        prepare_experimental_data,
        read_frequency_counts,
        resolve_context_col,
    )
except ImportError:
    from data_utils import (
        clean_word,
        normalize_unique_tokens,
        prepare_experimental_data,
        read_frequency_counts,
        resolve_context_col,
    )

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTAL_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_FREQUENCY_1GRAM_COUNTS = ROOT_DIR.parent / "ngrams" / "vocab_1gram_counts.tsv"
DEFAULT_FREQUENCY_2GRAM_COUNTS = ROOT_DIR.parent / "ngrams" / "vocab_2gram_counts.tsv"


def resolve_default_ordering_results() -> Path:
    candidates = [
        ROOT_DIR / "focus_alt_exp_pipeline" / "results" / "frequency" / "ordering_results_frequency.csv",
        ROOT_DIR / "focus_alt_exp_pipeline" / "results" / "ordering_results_frequency.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_ORDERING_RESULTS = resolve_default_ordering_results()


def _build_token_summary(
    experimental_data: pd.DataFrame,
    *,
    context: str,
    unigram_counts_path: Path,
    bigram_counts_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_experimental_data(experimental_data)
    context_col = resolve_context_col(prepared)
    context_df = prepared[prepared[context_col].astype(str) == context].copy()
    if context_df.empty:
        raise ValueError(f"Context '{context}' was not found in {context_col}")

    all_tokens = normalize_unique_tokens(
        pd.concat(
            [prepared["cleaned_query"], prepared["cleaned_trigger"]],
            ignore_index=True,
        ).tolist()
    )
    context_tokens = normalize_unique_tokens(
        pd.concat(
            [context_df["cleaned_query"], context_df["cleaned_trigger"]],
            ignore_index=True,
        ).tolist()
    )

    all_counts = read_frequency_counts(
        all_tokens,
        unigram_counts_path=unigram_counts_path,
        bigram_counts_path=bigram_counts_path,
    )
    total_supported_count = sum(all_counts.values())
    if total_supported_count <= 0:
        raise ValueError("No positive frequency counts were found for the experimental tokens")

    token_rows = []
    context_count_total = 0
    for token in context_tokens:
        count = all_counts.get(token)
        supported = count is not None and count > 0
        if supported:
            context_count_total += int(count)
        token_rows.append(
            {
                "context": context,
                "token": token,
                "count": int(count) if supported else pd.NA,
                "supported_by_frequency_model": bool(supported),
                "sampler_probability": (float(count) / total_supported_count) if supported else pd.NA,
            }
        )

    token_summary = pd.DataFrame(token_rows)
    if context_count_total > 0:
        token_summary["context_probability"] = token_summary["count"].apply(
            lambda value: (float(value) / context_count_total) if pd.notna(value) else pd.NA
        )
    else:
        token_summary["context_probability"] = pd.NA

    human_token_summary = (
        context_df.groupby("cleaned_query", as_index=False)
        .size()
        .rename(columns={"cleaned_query": "token", "size": "human_query_trials"})
    )
    human_trigger_summary = (
        context_df.groupby("cleaned_trigger", as_index=False)
        .size()
        .rename(columns={"cleaned_trigger": "token", "size": "human_trigger_trials"})
    )
    token_summary = token_summary.merge(human_token_summary, on="token", how="left")
    token_summary = token_summary.merge(human_trigger_summary, on="token", how="left")
    token_summary["human_query_trials"] = token_summary["human_query_trials"].fillna(0).astype(int)
    token_summary["human_trigger_trials"] = token_summary["human_trigger_trials"].fillna(0).astype(int)

    return token_summary, context_df


def _build_pairwise_summary(
    *,
    context: str,
    token_summary: pd.DataFrame,
    context_df: pd.DataFrame,
    ordering_results: pd.DataFrame,
) -> pd.DataFrame:
    available_tokens = token_summary[token_summary["supported_by_frequency_model"]].copy()
    if available_tokens.empty:
        raise ValueError(f"No supported frequency-model tokens were found for context '{context}'")

    token_lookup = available_tokens.set_index("token").to_dict(orient="index")
    pair_rows = []
    for query, trigger in permutations(available_tokens["token"].tolist(), 2):
        query_count = int(token_lookup[query]["count"])
        trigger_count = int(token_lookup[trigger]["count"])
        pair_rows.append(
            {
                "context": context,
                "query": query,
                "trigger": trigger,
                "query_count": query_count,
                "trigger_count": trigger_count,
                "query_sampler_probability": float(token_lookup[query]["sampler_probability"]),
                "trigger_sampler_probability": float(token_lookup[trigger]["sampler_probability"]),
                "query_context_probability": float(token_lookup[query]["context_probability"]),
                "trigger_context_probability": float(token_lookup[trigger]["context_probability"]),
                "prob_query_above_trigger_exact": query_count / float(query_count + trigger_count),
            }
        )

    pair_summary = pd.DataFrame(pair_rows)

    human_pair_summary = (
        context_df.groupby(["cleaned_query", "cleaned_trigger"], as_index=False)
        .agg(
            human_neg_rate=("neg", "mean"),
            human_trial_count=("neg", "size"),
        )
        .rename(columns={"cleaned_query": "query", "cleaned_trigger": "trigger"})
    )
    pair_summary = pair_summary.merge(human_pair_summary, on=["query", "trigger"], how="left")

    ordering_context = ordering_results[ordering_results["context"].astype(str) == context].copy()
    empirical_summary = (
        ordering_context.groupby(["query", "trigger"], as_index=False)
        .agg(
            prob_query_above_trigger_empirical=("negation_probability", "mean"),
            ordering_result_rows=("negation_probability", "size"),
            ordering_boundaries=("set_boundary", "nunique"),
        )
    )
    pair_summary = pair_summary.merge(empirical_summary, on=["query", "trigger"], how="left")
    pair_summary["empirical_minus_exact"] = (
        pair_summary["prob_query_above_trigger_empirical"]
        - pair_summary["prob_query_above_trigger_exact"]
    )

    return pair_summary.sort_values(
        by=["trigger", "query"],
        kind="stable",
        ignore_index=True,
    )


def _print_token_summary(token_summary: pd.DataFrame) -> None:
    display_cols = [
        "token",
        "count",
        "sampler_probability",
        "context_probability",
        "human_query_trials",
        "human_trigger_trials",
        "supported_by_frequency_model",
    ]
    print("\nToken summary")
    print(token_summary[display_cols].to_string(index=False))


def _print_pair_focus(pair_summary: pd.DataFrame, *, query: str, trigger: str) -> None:
    filtered = pair_summary[
        (pair_summary["query"] == clean_word(query))
        & (pair_summary["trigger"] == clean_word(trigger))
    ]
    if filtered.empty:
        raise ValueError(
            f"No pair summary row found for query='{query}' and trigger='{trigger}'"
        )

    display_cols = [
        "query",
        "trigger",
        "query_count",
        "trigger_count",
        "query_sampler_probability",
        "trigger_sampler_probability",
        "query_context_probability",
        "trigger_context_probability",
        "prob_query_above_trigger_exact",
        "prob_query_above_trigger_empirical",
        "empirical_minus_exact",
        "human_neg_rate",
        "human_trial_count",
        "ordering_result_rows",
        "ordering_boundaries",
    ]
    print("\nFocused pair summary")
    print(filtered[display_cols].to_string(index=False))


def _print_pair_preview(pair_summary: pd.DataFrame, limit: int) -> None:
    preview = pair_summary.sort_values(
        by=["prob_query_above_trigger_exact", "query", "trigger"],
        ascending=[False, True, True],
        kind="stable",
    ).head(limit)

    display_cols = [
        "query",
        "trigger",
        "query_count",
        "trigger_count",
        "prob_query_above_trigger_exact",
        "prob_query_above_trigger_empirical",
        "empirical_minus_exact",
        "human_neg_rate",
        "human_trial_count",
    ]
    print(f"\nTop {len(preview)} pair rows by exact query-above-trigger probability")
    print(preview[display_cols].to_string(index=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep-dive diagnostics for the frequency model in one context."
    )
    parser.add_argument("--context", type=str, required=True, help="Context name, for example 'bag'.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional query token to focus on.",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="Optional trigger token to focus on.",
    )
    parser.add_argument(
        "--experimental-data",
        type=Path,
        default=DEFAULT_EXPERIMENTAL_DATA,
        help="Human experimental CSV used to define contexts and tokens.",
    )
    parser.add_argument(
        "--ordering-results",
        type=Path,
        default=DEFAULT_ORDERING_RESULTS,
        help="Frequency ordering results CSV to compare against.",
    )
    parser.add_argument(
        "--frequency-1gram-counts",
        type=Path,
        default=DEFAULT_FREQUENCY_1GRAM_COUNTS,
        help="Unigram counts TSV used by the frequency model.",
    )
    parser.add_argument(
        "--frequency-2gram-counts",
        type=Path,
        default=DEFAULT_FREQUENCY_2GRAM_COUNTS,
        help="Bigram counts TSV used by the frequency model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write token and pairwise CSV summaries.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=12,
        help="How many pair rows to print in the default preview.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if (args.query is None) ^ (args.trigger is None):
        raise ValueError("--query and --trigger must be provided together")
    if args.preview_rows <= 0:
        raise ValueError("--preview-rows must be > 0")

    experimental_data = pd.read_csv(args.experimental_data)
    ordering_results = pd.read_csv(args.ordering_results)

    token_summary, context_df = _build_token_summary(
        experimental_data,
        context=args.context,
        unigram_counts_path=args.frequency_1gram_counts,
        bigram_counts_path=args.frequency_2gram_counts,
    )
    pair_summary = _build_pairwise_summary(
        context=args.context,
        token_summary=token_summary,
        context_df=context_df,
        ordering_results=ordering_results,
    )

    _print_token_summary(token_summary)
    if args.query is not None and args.trigger is not None:
        _print_pair_focus(pair_summary, query=args.query, trigger=args.trigger)
    else:
        _print_pair_preview(pair_summary, limit=args.preview_rows)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        safe_context = clean_word(args.context).replace(" ", "_")
        token_path = args.output_dir / f"{safe_context}_token_summary.csv"
        pair_path = args.output_dir / f"{safe_context}_pairwise_summary.csv"
        token_summary.to_csv(token_path, index=False)
        pair_summary.to_csv(pair_path, index=False)
        print(f"\nWrote {token_path}")
        print(f"Wrote {pair_path}")


if __name__ == "__main__":
    main()
