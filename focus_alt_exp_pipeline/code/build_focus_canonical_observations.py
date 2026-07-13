"""Build canonical observations for the novel focus-alternative dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .canonical_observations import (
        SCHEMA_VERSION,
        canonical_column_order,
        summarize_canonical_observations,
        validate_canonical_observations,
    )
except ImportError:
    from canonical_observations import (
        SCHEMA_VERSION,
        canonical_column_order,
        summarize_canonical_observations,
        validate_canonical_observations,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_SCORE_DATA = ROOT_DIR / "trigger_analysis" / "results" / "qwen_trigger_query_pair_logprobs.csv"
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "focus_alt_exp_pipeline"
    / "canonical_data"
    / "novel_focus_observations.csv"
)
UNKNOWN_REVISION = "unrecorded_existing_artifact"


def _portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(resolved)


def _require_columns(df: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _aggregate_human_responses(human_data: pd.DataFrame) -> pd.DataFrame:
    required = {"story", "cleaned_trigger", "cleaned_query", "neg"}
    _require_columns(human_data, required, label="Focus human data")

    prepared = human_data.loc[:, sorted(required)].copy()
    prepared["story"] = prepared["story"].astype(str).str.strip()
    prepared["cleaned_trigger"] = prepared["cleaned_trigger"].astype(str).str.strip()
    prepared["cleaned_query"] = prepared["cleaned_query"].astype(str).str.strip()
    prepared["neg"] = pd.to_numeric(prepared["neg"], errors="raise")
    if not prepared["neg"].isin([0, 1]).all():
        raise ValueError("Focus human neg responses must be binary 0/1 values")

    aggregated = (
        prepared.groupby(
            ["story", "cleaned_trigger", "cleaned_query"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            human_yes=("neg", "sum"),
            human_total=("neg", "size"),
            human_rate=("neg", "mean"),
        )
        .rename(
            columns={
                "cleaned_trigger": "trigger",
                "cleaned_query": "query",
            }
        )
    )
    aggregated["human_yes"] = aggregated["human_yes"].astype(int)
    aggregated["human_total"] = aggregated["human_total"].astype(int)
    return aggregated


def _prepare_pair_scores(score_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "story",
        "trigger",
        "query",
        "base_prefix",
        "trigger_prefix",
        "base_logprob_sum",
        "base_logprob_mean",
        "base_query_token_count",
        "base_tokenization_mode",
        "but_not_logprob_sum",
        "but_not_logprob_mean",
        "but_not_query_token_count",
        "but_not_tokenization_mode",
    }
    _require_columns(score_data, required, label="Focus Qwen score data")

    prepared = score_data.loc[:, sorted(required)].copy()
    for column in ["story", "trigger", "query"]:
        prepared[column] = prepared[column].astype(str).str.strip()

    key_columns = ["story", "trigger", "query"]
    if prepared.duplicated(key_columns).any():
        raise ValueError("Focus Qwen score data contains duplicate trigger/query keys")

    candidate_columns = [
        "base_prefix",
        "base_logprob_sum",
        "base_logprob_mean",
        "base_query_token_count",
        "base_tokenization_mode",
    ]
    consistency = prepared.groupby(["story", "query"], dropna=False)[candidate_columns].nunique(
        dropna=False
    )
    if (consistency > 1).any(axis=None):
        raise ValueError(
            "A focus candidate has inconsistent neutral-frame scores across trigger pairings"
        )

    candidate_scores = prepared.drop_duplicates(["story", "query"])[
        ["story", "query", *candidate_columns]
    ].copy()
    return prepared, candidate_scores


def build_focus_canonical_observations(
    human_data: pd.DataFrame,
    score_data: pd.DataFrame,
    *,
    model_name: str = "Qwen2-7B",
    model_revision: str = UNKNOWN_REVISION,
    source_human_file: str = "focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv",
    source_score_file: str = "trigger_analysis/results/qwen_trigger_query_pair_logprobs.csv",
) -> pd.DataFrame:
    """Join novel-focus human rates to neutral query and trigger Qwen scores."""

    human = _aggregate_human_responses(human_data)
    pair_scores, candidate_scores = _prepare_pair_scores(score_data)

    joined = human.merge(
        pair_scores,
        on=["story", "trigger", "query"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_query = joined["_merge"].ne("both")
    if missing_query.any():
        raise ValueError(
            f"Missing Qwen query scores for {int(missing_query.sum())} focus observations"
        )
    joined = joined.drop(columns="_merge")

    trigger_scores = candidate_scores.rename(
        columns={
            "query": "trigger",
            "base_prefix": "trigger_base_prefix",
            "base_logprob_sum": "trigger_logprob_sum",
            "base_logprob_mean": "trigger_logprob_mean",
            "base_query_token_count": "trigger_token_count",
            "base_tokenization_mode": "trigger_tokenization_mode",
        }
    )
    joined = joined.merge(
        trigger_scores,
        on=["story", "trigger"],
        how="left",
        validate="many_to_one",
    )
    if joined["trigger_logprob_sum"].isna().any():
        raise ValueError("Missing neutral-frame trigger scores for focus observations")
    if not joined["base_prefix"].eq(joined["trigger_base_prefix"]).all():
        raise ValueError("Focus trigger and query were not scored under the same base prompt")

    observations = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_family": "novel_focus",
            "dataset": "focus_alternative_study",
            "condition": "focus_only",
            "item_id": (
                joined["story"].astype(str)
                + "::"
                + joined["trigger"].astype(str)
                + "::"
                + joined["query"].astype(str)
            ),
            "group_id": joined["story"].astype(str),
            "context_id": joined["story"].astype(str),
            "generation_frame": "no_frame",
            "generation_prompt": joined["base_prefix"].astype(str),
            "trigger": joined["trigger"].astype(str),
            "query": joined["query"].astype(str),
            "human_outcome": "query_excluded",
            "human_yes": joined["human_yes"].astype(int),
            "human_total": joined["human_total"].astype(int),
            "human_rate": joined["human_rate"].astype(float),
            "trigger_logprob_sum": joined["trigger_logprob_sum"].astype(float),
            "query_logprob_sum": joined["base_logprob_sum"].astype(float),
            "trigger_logprob_mean": joined["trigger_logprob_mean"].astype(float),
            "query_logprob_mean": joined["base_logprob_mean"].astype(float),
            "trigger_token_count": joined["trigger_token_count"].astype(int),
            "query_token_count": joined["base_query_token_count"].astype(int),
            "trigger_tokenization_mode": joined["trigger_tokenization_mode"].astype(str),
            "query_tokenization_mode": joined["base_tokenization_mode"].astype(str),
            "x_but_not_y_applicable": True,
            "x_but_not_y_prompt": joined["trigger_prefix"].astype(str),
            "x_but_not_y_logprob_sum": joined["but_not_logprob_sum"].astype(float),
            "x_but_not_y_logprob_mean": joined["but_not_logprob_mean"].astype(float),
            "x_but_not_y_token_count": joined["but_not_query_token_count"].astype(int),
            "x_but_not_y_tokenization_mode": joined["but_not_tokenization_mode"].astype(str),
            "model_name": str(model_name),
            "model_revision": str(model_revision),
            "model_provenance_complete": str(model_revision) != UNKNOWN_REVISION,
            "source_human_file": str(source_human_file),
            "source_score_file": str(source_score_file),
        }
    )
    observations = observations.sort_values(
        ["context_id", "trigger", "query"],
        ignore_index=True,
    )
    observations = canonical_column_order(observations)
    validate_canonical_observations(observations)
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical observations for the novel focus dataset."
    )
    parser.add_argument("--human-data", type=Path, default=DEFAULT_HUMAN_DATA)
    parser.add_argument("--score-data", type=Path, default=DEFAULT_SCORE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", type=str, default="Qwen2-7B")
    parser.add_argument("--model-revision", type=str, default=UNKNOWN_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_data = pd.read_csv(args.human_data)
    score_data = pd.read_csv(args.score_data)
    observations = build_focus_canonical_observations(
        human_data,
        score_data,
        model_name=args.model_name,
        model_revision=args.model_revision,
        source_human_file=_portable_source_path(args.human_data),
        source_score_file=_portable_source_path(args.score_data),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output, index=False)
    summary = asdict(summarize_canonical_observations(observations))
    print(f"[complete] wrote {args.output}")
    for key, value in summary.items():
        print(f"  {key}={value}")
    if summary["incomplete_provenance_rows"]:
        print(
            "[warning] Existing focus score artifact does not record its resolved model revision."
        )


if __name__ == "__main__":
    main()
