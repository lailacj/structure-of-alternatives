"""Construct and attach query-only R&X contrastive scores without model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scoring_manifest import (
    FRAME_AWARE_MANIFEST_VERSION, MANIFEST_COLUMNS, SCORING_CONTROL_COLUMNS,
    validate_scoring_manifest,
)
from score_qwen_scoring_manifest import SCORING_BOUNDARY_VERSION, scoring_row_key
from standardized_score_adapters import score_provenance

PIPELINE = Path(__file__).resolve().parents[1]
REVISION = "453ed1575b739b5b03ce3758b23befdb0967f40e"
FAMILY = "ronai_xiang_2024"
CONDITIONS = ("ESI", "Eweak", "Estrong", "Eonly", "Eonlystrong")
DEFAULT_SOURCE = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15/inputs/source_rows.csv"
DEFAULT_MANIFEST = PIPELINE / "scoring_manifests/rnx_x_but_not_y_manifest.csv"
DEFAULT_SCORES = PIPELINE / "model_scores/rnx_x_but_not_y_qwen_scores.csv"
DEFAULT_PARENT_VIEWER = PIPELINE / "results_viewer/archive/corrected_pre_rnx_2026-10-26.html"
KEYS = ["dataset_family", "dataset", "condition", "context_id"]
FRAME_COLUMNS = [
    "x_but_not_y_applicable", "x_but_not_y_prompt", "x_but_not_y_logprob_sum",
    "x_but_not_y_logprob_mean", "x_but_not_y_token_count", "x_but_not_y_tokenization_mode",
    "x_but_not_y_trigger", "x_but_not_y_query", "x_but_not_y_score_artifact",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(source: pd.DataFrame, *, source_file: str) -> pd.DataFrame:
    """Use actual surface forms and QUD prefixes from the neutral source table."""
    rows = source.loc[source.dataset_family.eq(FAMILY)].copy()
    counts = rows.groupby("condition").size().to_dict()
    if counts != {condition: 60 for condition in CONDITIONS}:
        raise ValueError(f"Expected 60 R&X items in each of five conditions; found {counts}")
    if rows.duplicated(KEYS).any() or not rows.generation_frame.eq("no_frame").all():
        raise ValueError("Expected unique neutral R&X item-condition rows")
    item_groups = set(rows.loc[rows.condition.eq("ESI"), "group_id"])
    if len(item_groups) != 60 or any(
        set(group.group_id) != item_groups or group.group_id.nunique() != 60
        for _, group in rows.groupby("condition")
    ):
        raise ValueError("All five R&X conditions must contain the same 60 distinct item groups")
    manifest = rows[["dataset_family", "dataset", "condition", "item_id", "group_id", "context_id", "trigger", "query"]].copy()
    manifest["manifest_version"] = FRAME_AWARE_MANIFEST_VERSION
    manifest["source_row_id"] = rows.item_id.astype(str)
    manifest["item_id"] = rows.item_id.astype(str) + "::x_but_not_y"
    manifest["generation_frame"] = "x_but_not_y"
    manifest["generation_prompt"] = rows.generation_prompt.str.rstrip() + " " + rows.trigger.str.strip() + " but not "
    manifest["analysis_inclusion_status"] = "included"
    manifest["has_hu_test_suite"] = np.nan
    manifest["source_prompt_file"] = source_file
    manifest["prompt_provenance"] = "rnx_neutral_prefix_plus_weaker_surface_but_not;only_omitted;query_only"
    manifest["score_trigger"] = False
    manifest["score_query"] = True
    manifest = manifest.loc[:, [*MANIFEST_COLUMNS, *SCORING_CONTROL_COLUMNS]].sort_values(
        ["condition", "item_id"], ignore_index=True)
    validate_scoring_manifest(manifest)
    return manifest


def validate_scores(scores: pd.DataFrame, manifest: pd.DataFrame, *, manifest_sha256: str) -> None:
    """Fail on incomplete coverage, altered prompts, incompatible scoring or metadata."""
    validate_scoring_manifest(manifest)
    fields = [*MANIFEST_COLUMNS, "manifest_sha256", "model_revision", "scored_candidate_roles",
              "scoring_boundary_version", "scoring_row_key", "query_tokenization_mode",
              "query_token_count", "query_logprob_sum", "query_logprob_mean",
              "query_token_ids_json", "query_tokens_json", "query_token_logprobs_json",
              "trigger_logprob_sum", "trigger_logprob_mean", "trigger_token_count"]
    missing = set(fields).difference(scores.columns)
    if missing:
        raise ValueError(f"R&X framed scores are missing columns: {sorted(missing)}")
    if len(scores) != 300 or scores.duplicated(KEYS).any():
        raise ValueError("Expected exactly 300 unique R&X framed score rows")
    score_provenance(scores)
    expected = manifest.sort_values(KEYS).reset_index(drop=True)
    actual = scores.sort_values(KEYS).reset_index(drop=True)
    for column in MANIFEST_COLUMNS:
        if not actual[column].fillna("").astype(str).equals(expected[column].fillna("").astype(str)):
            raise ValueError(f"Framed scores do not match manifest column {column}")
    checks = {"manifest_sha256": manifest_sha256, "model_revision": REVISION,
              "scored_candidate_roles": "query_only", "scoring_boundary_version": SCORING_BOUNDARY_VERSION,
              "query_tokenization_mode": "exact_concat"}
    for column, value in checks.items():
        if not actual[column].eq(value).all():
            raise ValueError(f"Incompatible framed score {column}; expected {value}")
    for column in ("trigger_logprob_sum", "trigger_logprob_mean", "trigger_token_count"):
        if actual[column].notna().any():
            raise ValueError("R&X framed scoring must be query-only")
    if not actual.scoring_row_key.eq(actual.apply(scoring_row_key, axis=1)).all():
        raise ValueError("Framed scoring row keys disagree with their prompts/candidates")
    repeated = {}
    for row in actual.itertuples():
        ids = json.loads(row.query_token_ids_json)
        tokens = json.loads(row.query_tokens_json)
        logs = np.asarray(json.loads(row.query_token_logprobs_json), dtype=float)
        count = float(row.query_token_count)
        if (not count.is_integer() or count < 1 or not isinstance(ids, list) or not isinstance(tokens, list)
                or logs.ndim != 1 or len(ids) != count or len(tokens) != count or len(logs) != count
                or not all(isinstance(i, int) and i >= 0 for i in ids)
                or not np.isfinite(logs).all() or (logs > 0).any()):
            raise ValueError("Invalid framed token scores or token counts")
        if (not np.isclose(logs.sum(), row.query_logprob_sum, rtol=0, atol=1e-8)
                or not np.isclose(logs.mean(), row.query_logprob_mean, rtol=0, atol=1e-8)):
            raise ValueError("Framed token scores disagree with saved sum/mean")
        key = row.generation_prompt, row.query
        value = (ids, tokens, logs.tolist(), row.query_logprob_sum, row.query_logprob_mean)
        if key in repeated and repeated[key] != value:
            raise ValueError("Identical framed prompt/target pairs have conflicting scores")
        repeated[key] = value


def attach_scores(source: pd.DataFrame, scores: pd.DataFrame, *, artifact: str) -> pd.DataFrame:
    """Attach validated scores by item/condition, allowing only framed fields to change.

    Call validate_scores against the actual manifest first for artifact-level checks.
    Here the prompts and targets are independently checked against the parent inputs.
    """
    expected = build_manifest(source, source_file="parent-source")
    match_columns = [*KEYS, "source_row_id", "group_id", "trigger", "query", "generation_prompt"]
    actual = scores.sort_values(KEYS).reset_index(drop=True)
    for column in match_columns:
        if not actual[column].astype(str).equals(expected.sort_values(KEYS)[column].reset_index(drop=True).astype(str)):
            raise ValueError(f"Framed scores disagree with parent neutral inputs: {column}")
    rows = source.loc[source.dataset_family.eq(FAMILY)]
    if not rows.model_revision.eq(REVISION).all():
        raise ValueError("Parent R&X source uses a different model snapshot")
    lookup = scores.set_index(KEYS)
    attached = source.copy()
    mapping = {"x_but_not_y_prompt": "generation_prompt", "x_but_not_y_trigger": "trigger",
               "x_but_not_y_query": "query", "x_but_not_y_logprob_sum": "query_logprob_sum",
               "x_but_not_y_logprob_mean": "query_logprob_mean", "x_but_not_y_token_count": "query_token_count",
               "x_but_not_y_tokenization_mode": "query_tokenization_mode"}
    # Older canonical tables may not have the optional artifact/inflection columns.
    for column in FRAME_COLUMNS:
        if column not in attached:
            attached[column] = np.nan
        if column in ("x_but_not_y_prompt", "x_but_not_y_trigger", "x_but_not_y_query",
                      "x_but_not_y_tokenization_mode", "x_but_not_y_score_artifact"):
            attached[column] = attached[column].astype(object)
    for index, row in rows.iterrows():
        scored = lookup.loc[tuple(row[column] for column in KEYS)]
        attached.at[index, "x_but_not_y_applicable"] = True
        attached.at[index, "x_but_not_y_score_artifact"] = artifact
        for destination, origin in mapping.items():
            attached.at[index, destination] = scored[origin]
    preserved = [column for column in source if column not in FRAME_COLUMNS]
    pd.testing.assert_frame_equal(source[preserved], attached[preserved], check_exact=True)
    pd.testing.assert_frame_equal(source.loc[~source.dataset_family.eq(FAMILY)],
                                  attached.loc[~source.dataset_family.eq(FAMILY), source.columns], check_dtype=False)
    return attached
