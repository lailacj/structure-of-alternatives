"""Shared validation helpers for standardized Qwen score artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {
    "dataset_family",
    "dataset",
    "condition",
    "item_id",
    "group_id",
    "context_id",
    "generation_frame",
    "generation_prompt",
    "trigger",
    "query",
    "analysis_inclusion_status",
    "source_row_id",
    "trigger_logprob_sum",
    "trigger_logprob_mean",
    "trigger_token_count",
    "trigger_tokenization_mode",
    "query_logprob_sum",
    "query_logprob_mean",
    "query_token_count",
    "query_tokenization_mode",
    "model_identifier",
    "model_revision",
    "model_revision_source",
    "resolved_model_path",
    "actual_model_dtype",
    "torch_version",
    "transformers_version",
}


@dataclass(frozen=True)
class ScoreProvenance:
    model_name: str
    model_revision: str


def require_columns(df: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def select_score_rows(
    scores: pd.DataFrame,
    *,
    dataset_family: str,
    generation_frame: str,
    expected_rows: int,
    label: str,
) -> pd.DataFrame:
    """Select one score family/frame and validate its basic scoring grain."""

    require_columns(scores, REQUIRED_SCORE_COLUMNS, label=label)
    selected = scores.loc[
        scores["dataset_family"].astype(str).eq(dataset_family)
        & scores["generation_frame"].astype(str).eq(generation_frame)
    ].copy()
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} {label} rows, found {len(selected)}")
    if selected["source_row_id"].isna().any():
        raise ValueError(f"{label} contains empty source_row_id values")
    if selected.duplicated(["dataset", "condition", "source_row_id"]).any():
        raise ValueError(f"{label} contains duplicate dataset/condition/source rows")
    if selected["query_logprob_sum"].isna().any():
        raise ValueError(f"{label} contains missing query scores")
    if generation_frame == "no_frame" and selected["trigger_logprob_sum"].isna().any():
        raise ValueError(f"{label} contains missing trigger scores")
    if generation_frame == "x_but_not_y" and selected["trigger_logprob_sum"].notna().any():
        raise ValueError(f"{label} must be query-only but contains trigger scores")
    return selected


def score_provenance(*score_frames: pd.DataFrame) -> ScoreProvenance:
    """Require one fully specified model snapshot across score artifacts."""

    if not score_frames:
        raise ValueError("At least one score frame is required")
    combined = pd.concat(score_frames, ignore_index=True)
    fields = [
        "model_identifier",
        "model_revision",
        "model_revision_source",
        "resolved_model_path",
        "actual_model_dtype",
        "torch_version",
        "transformers_version",
    ]
    for field in fields:
        values = combined[field].dropna().astype(str).str.strip().unique().tolist()
        if len(values) != 1 or values[0] == "":
            raise ValueError(
                f"Standardized score rows must have one non-empty {field}; found {values}"
            )

    revision = str(combined["model_revision"].iloc[0]).strip()
    resolved_path = str(combined["resolved_model_path"].iloc[0])
    if revision not in resolved_path:
        raise ValueError("Resolved Qwen model path does not contain the recorded revision")
    if not np.isfinite(
        pd.to_numeric(combined["query_logprob_sum"], errors="coerce").to_numpy(dtype=float)
    ).all():
        raise ValueError("Standardized score rows contain non-finite query scores")
    return ScoreProvenance(model_name="Qwen2-7B", model_revision=revision)


def require_matching_candidates(
    joined: pd.DataFrame,
    *,
    left_trigger: str,
    left_query: str,
    right_trigger: str,
    right_query: str,
    label: str,
) -> None:
    """Check that a human/model join did not align different candidates."""

    def normalized(column: str) -> pd.Series:
        return joined[column].astype(str).str.strip().str.lower()

    mismatch = normalized(left_trigger).ne(normalized(right_trigger)) | normalized(
        left_query
    ).ne(normalized(right_query))
    if mismatch.any():
        sample = joined.loc[
            mismatch,
            [left_trigger, left_query, right_trigger, right_query],
        ].head(3)
        raise ValueError(
            f"{label} joined rows contain candidate mismatches: {sample.to_dict('records')}"
        )
