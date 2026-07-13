"""Build canonical observations for the novel focus-alternative dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

try:
    from .canonical_observations import (
        SCHEMA_VERSION,
        canonical_column_order,
        summarize_canonical_observations,
        validate_canonical_observations,
    )
    from .standardized_score_adapters import (
        require_columns,
        require_matching_candidates,
        score_provenance,
        select_score_rows,
    )
except ImportError:
    from canonical_observations import (
        SCHEMA_VERSION,
        canonical_column_order,
        summarize_canonical_observations,
        validate_canonical_observations,
    )
    from standardized_score_adapters import (
        require_columns,
        require_matching_candidates,
        score_provenance,
        select_score_rows,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_SCORE_DATA = (
    ROOT_DIR
    / "focus_alt_exp_pipeline"
    / "model_scores"
    / "focus_hu_remaining_qwen_scores.csv"
)
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "focus_alt_exp_pipeline"
    / "canonical_data"
    / "novel_focus_observations.csv"
)


def _portable_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(resolved)


def _aggregate_human_responses(
    human_data: pd.DataFrame,
    *,
    expected_rows: int,
) -> pd.DataFrame:
    required = {"story", "cleaned_trigger", "cleaned_query", "neg"}
    require_columns(human_data, required, label="Focus human data")

    prepared = human_data.loc[:, sorted(required)].copy()
    prepared["story"] = prepared["story"].astype(str).str.strip()
    prepared["trigger"] = prepared.pop("cleaned_trigger").astype(str).str.strip().str.lower()
    prepared["query"] = prepared.pop("cleaned_query").astype(str).str.strip().str.lower()
    prepared["neg"] = pd.to_numeric(prepared["neg"], errors="raise")
    if not prepared["neg"].isin([0, 1]).all():
        raise ValueError("Focus human neg responses must be binary 0/1 values")

    aggregated = (
        prepared.groupby(
            ["story", "trigger", "query"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            human_yes=("neg", "sum"),
            human_total=("neg", "size"),
            human_rate=("neg", "mean"),
        )
    )
    aggregated["human_yes"] = aggregated["human_yes"].astype(int)
    aggregated["human_total"] = aggregated["human_total"].astype(int)
    aggregated["source_row_id"] = (
        aggregated["story"] + "::" + aggregated["trigger"] + "::" + aggregated["query"]
    )
    if len(aggregated) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} focus observations, found {len(aggregated)}"
        )
    return aggregated


def build_focus_canonical_observations(
    human_data: pd.DataFrame,
    score_data: pd.DataFrame,
    *,
    source_human_file: str = "focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv",
    source_score_file: str = (
        "focus_alt_exp_pipeline/model_scores/focus_hu_remaining_qwen_scores.csv"
    ),
    expected_rows: int = 480,
) -> pd.DataFrame:
    """Join focus human counts to pinned no-frame and X-but-not-Y scores."""

    human = _aggregate_human_responses(human_data, expected_rows=expected_rows)
    no_frame = select_score_rows(
        score_data,
        dataset_family="novel_focus",
        generation_frame="no_frame",
        expected_rows=expected_rows,
        label="focus no-frame scores",
    )
    x_frame = select_score_rows(
        score_data,
        dataset_family="novel_focus",
        generation_frame="x_but_not_y",
        expected_rows=expected_rows,
        label="focus X-but-not-Y scores",
    )
    provenance = score_provenance(no_frame, x_frame)

    no_columns = [
        "source_row_id",
        "group_id",
        "context_id",
        "generation_prompt",
        "trigger",
        "query",
        "trigger_logprob_sum",
        "query_logprob_sum",
        "trigger_logprob_mean",
        "query_logprob_mean",
        "trigger_token_count",
        "query_token_count",
        "trigger_tokenization_mode",
        "query_tokenization_mode",
    ]
    joined = human.merge(
        no_frame.loc[:, no_columns],
        on="source_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_human", "_score"),
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Some focus human observations are missing no-frame scores")
    joined = joined.drop(columns="_merge")
    require_matching_candidates(
        joined,
        left_trigger="trigger_human",
        left_query="query_human",
        right_trigger="trigger_score",
        right_query="query_score",
        label="Focus no-frame",
    )

    x_columns = [
        "source_row_id",
        "generation_prompt",
        "trigger",
        "query",
        "query_logprob_sum",
        "query_logprob_mean",
        "query_token_count",
        "query_tokenization_mode",
    ]
    x_scores = x_frame.loc[:, x_columns].rename(
        columns={
            column: f"{column}_x"
            for column in x_columns
            if column != "source_row_id"
        }
    )
    joined = joined.merge(
        x_scores,
        on="source_row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Some focus human observations are missing X-but-not-Y scores")
    joined = joined.drop(columns="_merge")
    require_matching_candidates(
        joined,
        left_trigger="trigger_human",
        left_query="query_human",
        right_trigger="trigger_x",
        right_query="query_x",
        label="Focus X-but-not-Y",
    )

    observations = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_family": "novel_focus",
            "dataset": "focus_alternative_study",
            "condition": "focus_only",
            "item_id": joined["source_row_id"].astype(str),
            "group_id": joined["group_id"].astype(str),
            "context_id": joined["context_id"].astype(str),
            "generation_frame": "no_frame",
            "generation_prompt": joined["generation_prompt"].astype(str),
            "trigger": joined["trigger_human"].astype(str),
            "query": joined["query_human"].astype(str),
            "human_outcome": "query_excluded",
            "human_yes": joined["human_yes"].astype(int),
            "human_total": joined["human_total"].astype(int),
            "human_rate": joined["human_rate"].astype(float),
            "human_count_status": "exact",
            "trigger_logprob_sum": joined["trigger_logprob_sum"].astype(float),
            "query_logprob_sum": joined["query_logprob_sum"].astype(float),
            "trigger_logprob_mean": joined["trigger_logprob_mean"].astype(float),
            "query_logprob_mean": joined["query_logprob_mean"].astype(float),
            "trigger_token_count": joined["trigger_token_count"].astype(int),
            "query_token_count": joined["query_token_count"].astype(int),
            "trigger_tokenization_mode": joined["trigger_tokenization_mode"].astype(str),
            "query_tokenization_mode": joined["query_tokenization_mode"].astype(str),
            "x_but_not_y_applicable": True,
            "x_but_not_y_prompt": joined["generation_prompt_x"].astype(str),
            "x_but_not_y_logprob_sum": joined["query_logprob_sum_x"].astype(float),
            "x_but_not_y_logprob_mean": joined["query_logprob_mean_x"].astype(float),
            "x_but_not_y_token_count": joined["query_token_count_x"].astype(int),
            "x_but_not_y_tokenization_mode": joined[
                "query_tokenization_mode_x"
            ].astype(str),
            "model_name": provenance.model_name,
            "model_revision": provenance.model_revision,
            "model_provenance_complete": True,
            "source_human_file": str(source_human_file),
            "source_score_file": str(source_score_file),
        }
    )
    observations = observations.sort_values(
        ["context_id", "trigger", "query"],
        ignore_index=True,
    )
    observations = canonical_column_order(observations)
    validate_canonical_observations(
        observations,
        require_complete_provenance=True,
        require_human_counts=True,
    )
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical observations for the novel focus dataset."
    )
    parser.add_argument("--human-data", type=Path, default=DEFAULT_HUMAN_DATA)
    parser.add_argument("--score-data", type=Path, default=DEFAULT_SCORE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = build_focus_canonical_observations(
        pd.read_csv(args.human_data),
        pd.read_csv(args.score_data),
        source_human_file=_portable_source_path(args.human_data),
        source_score_file=_portable_source_path(args.score_data),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output, index=False)
    print(f"[complete] wrote {args.output}")
    for key, value in asdict(summarize_canonical_observations(observations)).items():
        print(f"  {key}={value}")


if __name__ == "__main__":
    main()
