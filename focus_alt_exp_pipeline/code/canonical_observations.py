"""Canonical cross-dataset observation schema and validation.

The canonical table separates data preparation from model evaluation.  Each
row represents one item-condition observation with an uttered trigger, a
queried alternative, aggregated human exclusion responses, and Qwen scores for
both candidates in one neutral no-frame generation prompt.

No predicted alternative-structure probabilities belong in this table.  Those
are derived later from the query and trigger scores plus parameters learned
inside cross-validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


SCHEMA_VERSION: Final[str] = "1.0"

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
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
    "human_outcome",
    "human_yes",
    "human_total",
    "human_rate",
    "trigger_logprob_sum",
    "query_logprob_sum",
    "trigger_logprob_mean",
    "query_logprob_mean",
    "trigger_token_count",
    "query_token_count",
    "trigger_tokenization_mode",
    "query_tokenization_mode",
    "x_but_not_y_applicable",
    "x_but_not_y_prompt",
    "x_but_not_y_logprob_sum",
    "x_but_not_y_logprob_mean",
    "x_but_not_y_token_count",
    "x_but_not_y_tokenization_mode",
    "model_name",
    "model_revision",
    "model_provenance_complete",
    "source_human_file",
    "source_score_file",
)

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset_family",
    "dataset",
    "condition",
    "item_id",
)

NONEMPTY_STRING_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
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
    "human_outcome",
    "trigger_tokenization_mode",
    "query_tokenization_mode",
    "model_name",
    "model_revision",
    "source_human_file",
    "source_score_file",
)


@dataclass(frozen=True)
class CanonicalObservationSummary:
    row_count: int
    dataset_family_count: int
    dataset_count: int
    condition_count: int
    group_count: int
    incomplete_provenance_rows: int
    x_but_not_y_rows: int


def _require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Canonical observations are missing columns: {missing}")


def _require_nonempty_strings(df: pd.DataFrame) -> None:
    for column in NONEMPTY_STRING_COLUMNS:
        values = df[column]
        invalid = values.isna() | values.astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(
                f"Canonical column '{column}' contains {int(invalid.sum())} empty values"
            )


def _require_integer_values(values: pd.Series, *, column: str, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | ~np.isclose(numeric, np.round(numeric)) | (numeric < minimum)
    if invalid.any():
        raise ValueError(
            f"Canonical column '{column}' contains {int(invalid.sum())} invalid integer values"
        )
    return numeric.astype(np.int64)


def _require_finite(values: pd.Series, *, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid.any():
        raise ValueError(
            f"Canonical column '{column}' contains {int(invalid.sum())} non-finite values"
        )
    return numeric.astype(float)


def _validate_logprob_family(
    df: pd.DataFrame,
    *,
    prefix: str,
    mask: pd.Series | None = None,
) -> None:
    selected = df if mask is None else df.loc[mask]
    if selected.empty:
        return

    summed = _require_finite(selected[f"{prefix}_logprob_sum"], column=f"{prefix}_logprob_sum")
    mean = _require_finite(
        selected[f"{prefix}_logprob_mean"],
        column=f"{prefix}_logprob_mean",
    )
    token_count = _require_integer_values(
        selected[f"{prefix}_token_count"],
        column=f"{prefix}_token_count",
        minimum=1,
    )
    expected_mean = summed / token_count
    if not np.allclose(mean, expected_mean, rtol=1e-8, atol=1e-8):
        raise ValueError(
            f"Canonical {prefix} mean log probabilities do not equal sum / token_count"
        )
    if (summed > 1e-10).any():
        raise ValueError(f"Canonical {prefix} log probabilities must be non-positive")


def validate_canonical_observations(
    observations: pd.DataFrame,
    *,
    require_complete_provenance: bool = False,
) -> None:
    """Raise ``ValueError`` if a canonical observation table is inconsistent."""

    if observations.empty:
        raise ValueError("Canonical observations must contain at least one row")

    _require_columns(observations)
    df = observations.loc[:, CANONICAL_COLUMNS].copy()
    _require_nonempty_strings(df)

    if not df["schema_version"].astype(str).eq(SCHEMA_VERSION).all():
        raise ValueError(f"All rows must use canonical schema version {SCHEMA_VERSION}")

    duplicate_keys = df.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate_keys.any():
        raise ValueError(
            f"Canonical observations contain {int(duplicate_keys.sum())} duplicate key rows"
        )

    human_yes = _require_integer_values(df["human_yes"], column="human_yes", minimum=0)
    human_total = _require_integer_values(
        df["human_total"],
        column="human_total",
        minimum=1,
    )
    if (human_yes > human_total).any():
        raise ValueError("Canonical human_yes cannot exceed human_total")

    human_rate = _require_finite(df["human_rate"], column="human_rate")
    expected_rate = human_yes / human_total
    if not np.allclose(human_rate, expected_rate, rtol=1e-10, atol=1e-10):
        raise ValueError("Canonical human_rate does not equal human_yes / human_total")
    if ((human_rate < 0.0) | (human_rate > 1.0)).any():
        raise ValueError("Canonical human_rate must be between zero and one")

    _validate_logprob_family(df, prefix="trigger")
    _validate_logprob_family(df, prefix="query")

    applicable = df["x_but_not_y_applicable"]
    if not applicable.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Canonical x_but_not_y_applicable values must be booleans")

    applicable = applicable.astype(bool)
    if applicable.any():
        x_rows = df.loc[applicable]
        empty_prompt = x_rows["x_but_not_y_prompt"].isna() | x_rows[
            "x_but_not_y_prompt"
        ].astype(str).str.strip().eq("")
        empty_mode = x_rows["x_but_not_y_tokenization_mode"].isna() | x_rows[
            "x_but_not_y_tokenization_mode"
        ].astype(str).str.strip().eq("")
        if empty_prompt.any() or empty_mode.any():
            raise ValueError("Applicable X-but-not-Y rows require a prompt and tokenization mode")
        _validate_logprob_family(df, prefix="x_but_not_y", mask=applicable)

    if (~applicable).any():
        nonapplicable = df.loc[~applicable]
        score_columns = [
            "x_but_not_y_logprob_sum",
            "x_but_not_y_logprob_mean",
            "x_but_not_y_token_count",
        ]
        if nonapplicable[score_columns].notna().any(axis=None):
            raise ValueError("Non-applicable X-but-not-Y rows must not contain scores")

    provenance_complete = df["model_provenance_complete"]
    if not provenance_complete.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Canonical model_provenance_complete values must be booleans")
    if require_complete_provenance and not provenance_complete.astype(bool).all():
        raise ValueError("Canonical observations contain incomplete model provenance")


def summarize_canonical_observations(
    observations: pd.DataFrame,
) -> CanonicalObservationSummary:
    validate_canonical_observations(observations)
    return CanonicalObservationSummary(
        row_count=len(observations),
        dataset_family_count=observations["dataset_family"].nunique(),
        dataset_count=observations["dataset"].nunique(),
        condition_count=observations["condition"].nunique(),
        group_count=observations["group_id"].nunique(),
        incomplete_provenance_rows=int(
            (~observations["model_provenance_complete"].astype(bool)).sum()
        ),
        x_but_not_y_rows=int(observations["x_but_not_y_applicable"].astype(bool).sum()),
    )


def canonical_column_order(observations: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical columns in their stable storage order."""

    _require_columns(observations)
    extras = [column for column in observations.columns if column not in CANONICAL_COLUMNS]
    return observations.loc[:, [*CANONICAL_COLUMNS, *extras]].copy()
