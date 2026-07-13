"""Schema and validation for cross-dataset Qwen scoring manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


MANIFEST_VERSION: Final[str] = "1.0"
FRAME_AWARE_MANIFEST_VERSION: Final[str] = "1.1"
SUPPORTED_MANIFEST_VERSIONS: Final[frozenset[str]] = frozenset(
    {MANIFEST_VERSION, FRAME_AWARE_MANIFEST_VERSION}
)
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "manifest_version",
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
    "has_hu_test_suite",
    "source_prompt_file",
    "source_row_id",
    "prompt_provenance",
)

MANIFEST_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset_family",
    "dataset",
    "condition",
    "item_id",
)

SCORING_CONTROL_COLUMNS: Final[tuple[str, ...]] = (
    "score_trigger",
    "score_query",
)

NONEMPTY_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in MANIFEST_COLUMNS if column != "has_hu_test_suite"
)


@dataclass(frozen=True)
class ScoringManifestSummary:
    row_count: int
    dataset_family_count: int
    dataset_count: int
    condition_count: int
    generation_frame_count: int
    unique_prompt_count: int
    unique_prompt_candidate_count: int
    hu_rows_pending_exact_filter: int


def _boolean_flag(values: pd.Series, *, column: str) -> pd.Series:
    normalized = values.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower()
    )
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    invalid = ~normalized.isin(mapping)
    if invalid.any():
        examples = sorted(set(normalized.loc[invalid].astype(str)))[:5]
        raise ValueError(
            f"Scoring manifest column '{column}' contains invalid boolean values: {examples}"
        )
    return normalized.map(mapping).astype(bool)


def scoring_candidate_flags(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return per-row trigger/query score flags, defaulting v1.0 rows to both."""

    present = [column in manifest.columns for column in SCORING_CONTROL_COLUMNS]
    if any(present) and not all(present):
        raise ValueError(
            "Scoring manifest must include both score_trigger and score_query together"
        )
    if not any(present):
        return pd.DataFrame(
            {
                "score_trigger": True,
                "score_query": True,
            },
            index=manifest.index,
        )
    return pd.DataFrame(
        {
            column: _boolean_flag(manifest[column], column=column)
            for column in SCORING_CONTROL_COLUMNS
        },
        index=manifest.index,
    )


def scored_prompt_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return the unique prompt-candidate pairs the scorer will evaluate."""

    flags = scoring_candidate_flags(manifest)
    parts: list[pd.DataFrame] = []
    for role in ["trigger", "query"]:
        selected = flags[f"score_{role}"]
        if selected.any():
            part = manifest.loc[selected, ["generation_prompt", role]].rename(
                columns={role: "candidate"}
            )
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["generation_prompt", "candidate"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)


def validate_scoring_manifest(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        raise ValueError("Scoring manifest must contain at least one row")

    missing = [column for column in MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Scoring manifest is missing columns: {missing}")

    df = manifest.loc[:, MANIFEST_COLUMNS].copy()
    for column in NONEMPTY_COLUMNS:
        invalid = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(
                f"Scoring manifest column '{column}' contains {int(invalid.sum())} empty values"
            )

    versions = set(df["manifest_version"].astype(str))
    unsupported = versions.difference(SUPPORTED_MANIFEST_VERSIONS)
    if unsupported:
        raise ValueError(f"Unsupported scoring manifest versions: {sorted(unsupported)}")
    if len(versions) != 1:
        raise ValueError("A scoring manifest must use one schema version consistently")
    version = next(iter(versions))
    flags = scoring_candidate_flags(manifest)
    if version == FRAME_AWARE_MANIFEST_VERSION and not all(
        column in manifest.columns for column in SCORING_CONTROL_COLUMNS
    ):
        raise ValueError(
            f"Manifest version {FRAME_AWARE_MANIFEST_VERSION} requires "
            "score_trigger and score_query columns"
        )
    frames = df["generation_frame"].astype(str)
    invalid_frames = ~frames.isin(["no_frame", "x_but_not_y"])
    if invalid_frames.any():
        raise ValueError(
            "generation_frame must be either 'no_frame' or 'x_but_not_y'"
        )
    if version == MANIFEST_VERSION and not frames.eq("no_frame").all():
        raise ValueError("Manifest version 1.0 supports only generation_frame='no_frame'")
    if (~flags["score_query"]).any():
        raise ValueError("Every scoring row must score the query candidate")
    no_frame = frames.eq("no_frame")
    x_but_not_y = frames.eq("x_but_not_y")
    if (~flags.loc[no_frame, "score_trigger"]).any():
        raise ValueError("No-frame rows must score both trigger and query candidates")
    if flags.loc[x_but_not_y, "score_trigger"].any():
        raise ValueError("X-but-not-Y rows must score the query only")
    unsupported_x_frame = x_but_not_y & ~df["dataset_family"].astype(str).isin(
        ["hu_2023_benchmark", "novel_focus"]
    )
    if unsupported_x_frame.any():
        raise ValueError("X-but-not-Y scoring applies only to Hu and novel-focus rows")
    if df.duplicated(list(MANIFEST_KEY_COLUMNS)).any():
        raise ValueError("Scoring manifest contains duplicate item-condition keys")
    if df["trigger"].astype(str).eq(df["query"].astype(str)).any():
        raise ValueError("Scoring manifest trigger and query must differ")
    if not df["generation_prompt"].astype(str).str[-1].str.isspace().all():
        raise ValueError("Every generation prompt must end in whitespace before the candidate")

    rnx = df["dataset_family"].astype(str).eq("ronai_xiang_2024")
    if rnx.any():
        answer_prefix = df.loc[rnx, "generation_prompt"].astype(str).str.rsplit("\n", n=1).str[-1]
        contains_only = answer_prefix.str.contains(r"\bonly\b", case=False, regex=True)
        if contains_only.any():
            raise ValueError("R&X no-frame answer prompts must not contain the word 'only'")

        rnx_rows = df.loc[rnx]
        for left_condition, right_condition in [
            ("ESI", "Eonly"),
            ("Estrong", "Eonlystrong"),
        ]:
            left = rnx_rows.loc[
                rnx_rows["condition"].eq(left_condition),
                ["group_id", "generation_prompt", "trigger", "query"],
            ].set_index("group_id")
            right = rnx_rows.loc[
                rnx_rows["condition"].eq(right_condition),
                ["group_id", "generation_prompt", "trigger", "query"],
            ].set_index("group_id")
            if set(left.index) != set(right.index):
                raise ValueError(
                    f"R&X {left_condition}/{right_condition} item coverage does not match"
                )
            if not left.sort_index().equals(right.sort_index()):
                raise ValueError(
                    f"R&X {left_condition}/{right_condition} no-frame prompts must match exactly"
                )


def summarize_scoring_manifest(manifest: pd.DataFrame) -> ScoringManifestSummary:
    validate_scoring_manifest(manifest)
    candidate_rows = scored_prompt_candidates(manifest)
    return ScoringManifestSummary(
        row_count=len(manifest),
        dataset_family_count=manifest["dataset_family"].nunique(),
        dataset_count=manifest["dataset"].nunique(),
        condition_count=manifest["condition"].nunique(),
        generation_frame_count=manifest["generation_frame"].nunique(),
        unique_prompt_count=manifest["generation_prompt"].nunique(),
        unique_prompt_candidate_count=len(candidate_rows),
        hu_rows_pending_exact_filter=int(
            manifest["analysis_inclusion_status"].astype(str).str.startswith("pending_hu").sum()
        ),
    )
