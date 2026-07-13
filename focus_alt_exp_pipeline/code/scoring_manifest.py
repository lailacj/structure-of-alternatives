"""Schema and validation for cross-dataset no-frame Qwen scoring manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


MANIFEST_VERSION: Final[str] = "1.0"
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

NONEMPTY_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in MANIFEST_COLUMNS if column != "has_hu_test_suite"
)


@dataclass(frozen=True)
class ScoringManifestSummary:
    row_count: int
    dataset_family_count: int
    dataset_count: int
    condition_count: int
    unique_prompt_count: int
    unique_prompt_candidate_count: int
    hu_rows_pending_exact_filter: int


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

    if not df["manifest_version"].astype(str).eq(MANIFEST_VERSION).all():
        raise ValueError(f"All scoring rows must use manifest version {MANIFEST_VERSION}")
    if not df["generation_frame"].astype(str).eq("no_frame").all():
        raise ValueError("Every scoring row must use generation_frame='no_frame'")
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
    candidate_rows = pd.concat(
        [
            manifest[["generation_prompt", "trigger"]].rename(columns={"trigger": "candidate"}),
            manifest[["generation_prompt", "query"]].rename(columns={"query": "candidate"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    return ScoringManifestSummary(
        row_count=len(manifest),
        dataset_family_count=manifest["dataset_family"].nunique(),
        dataset_count=manifest["dataset"].nunique(),
        condition_count=manifest["condition"].nunique(),
        unique_prompt_count=manifest["generation_prompt"].nunique(),
        unique_prompt_candidate_count=len(candidate_rows),
        hu_rows_pending_exact_filter=int(
            manifest["analysis_inclusion_status"].astype(str).str.startswith("pending_hu").sum()
        ),
    )
