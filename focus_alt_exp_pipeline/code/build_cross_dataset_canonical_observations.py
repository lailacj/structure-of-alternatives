"""Build Hu, Ronai-Xiang, and combined canonical observation tables."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .build_focus_canonical_observations import build_focus_canonical_observations
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
    from build_focus_canonical_observations import build_focus_canonical_observations
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


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EXPERIMENT_ROOT = WORKSPACE_ROOT / "experiment_ronai&xiang"
PIPELINE_ROOT = REPO_ROOT / "focus_alt_exp_pipeline"

DEFAULT_FOCUS_HUMAN = PIPELINE_ROOT / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_NO_FRAME_SCORES = PIPELINE_ROOT / "model_scores" / "hu_rnx_no_frame_qwen_scores.csv"
DEFAULT_REMAINING_SCORES = (
    PIPELINE_ROOT / "model_scores" / "focus_hu_remaining_qwen_scores.csv"
)
DEFAULT_HU_HUMAN = (
    EXPERIMENT_ROOT / "jen_hu_modeling" / "data_processed" / "hu_cross_scale_items.csv"
)
DEFAULT_HU_RX22_TRIALS = (
    EXPERIMENT_ROOT
    / "jen_hu_modeling"
    / "data_raw"
    / "hu_et_al_2023"
    / "cross-scale"
    / "human_data"
    / "rx22_processed_exp1.csv"
)
DEFAULT_RNX_HUMAN = (
    EXPERIMENT_ROOT
    / "human_model_analysis"
    / "human_response_rates_by_item_condition.csv"
)
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "canonical_data"

# Hu et al.'s cross-scale notebook joins human rows to GPT-2 results and calls
# bare ``dropna()`` before the Figure 3 correlations.  At source commit
# 50a7064290a841b81b2608524000b81a33ddc4b0, the following scales lack the
# string-based strong-scalemate surprisal.  The same call also removes vt16's
# unsettling/horrific scale because its unrelated LSA column is empty.  We
# reproduce that literal published-code subset here.  The notebook then
# averages the three vt16 sentence templates to one scale-level score.
HU_ORIGINAL_MISSING_STRING_SURPRISAL = {
    "g18": {"transparent/crystal clear", "unkind/nasty", "sick/terminally ill"},
    "pvt21": set(),
    "rx22": {"damage/destroy", "or/and", "some/all"},
    "vt16": {"few/none", "may/have to", "some/all"},
}
HU_ORIGINAL_OTHER_DROPNA_EXCLUSIONS = {
    "g18": set(),
    "pvt21": set(),
    "rx22": set(),
    "vt16": {"unsettling/horrific"},
}
HU_ORIGINAL_INCLUDED_PROMPT_ROWS = {"g18": 67, "pvt21": 50, "rx22": 57, "vt16": 117}
HU_ORIGINAL_INCLUDED_SCALES = {"g18": 67, "pvt21": 50, "rx22": 57, "vt16": 39}


def _source_path(path: Path) -> str:
    resolved = path.resolve()
    for root in (REPO_ROOT, WORKSPACE_ROOT):
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(resolved)


def _prepare_hu_noframe(scores: pd.DataFrame) -> pd.DataFrame:
    selected = select_score_rows(
        scores,
        dataset_family="hu_2023_benchmark",
        generation_frame="no_frame",
        expected_rows=309,
        label="Hu no-frame scores",
    )
    columns = [
        "dataset",
        "context_id",
        "group_id",
        "generation_prompt",
        "trigger",
        "query",
        "analysis_inclusion_status",
        "has_hu_test_suite",
        "trigger_logprob_sum",
        "query_logprob_sum",
        "trigger_logprob_mean",
        "query_logprob_mean",
        "trigger_token_count",
        "query_token_count",
        "trigger_tokenization_mode",
        "query_tokenization_mode",
    ]
    prepared = selected.loc[:, columns].rename(
        columns={
            "context_id": "human_item_id",
            "trigger": "trigger_score",
            "query": "query_score",
        }
    )
    return prepared


def _prepare_hu_xframe(scores: pd.DataFrame) -> pd.DataFrame:
    selected = select_score_rows(
        scores,
        dataset_family="hu_2023_benchmark",
        generation_frame="x_but_not_y",
        expected_rows=309,
        label="Hu X-but-not-Y scores",
    )
    return selected.loc[
        :,
        [
            "dataset",
            "context_id",
            "group_id",
            "generation_prompt",
            "trigger",
            "query",
            "query_logprob_sum",
            "query_logprob_mean",
            "query_token_count",
            "query_tokenization_mode",
        ],
    ].rename(
        columns={
            "context_id": "human_item_id",
            "group_id": "group_id_x",
            "generation_prompt": "x_but_not_y_prompt",
            "trigger": "trigger_x",
            "query": "query_x",
            "query_logprob_sum": "x_but_not_y_logprob_sum",
            "query_logprob_mean": "x_but_not_y_logprob_mean",
            "query_token_count": "x_but_not_y_token_count",
            "query_tokenization_mode": "x_but_not_y_tokenization_mode",
        }
    )


def _rx22_counts(rx22_trials: pd.DataFrame) -> pd.DataFrame:
    require_columns(rx22_trials, {"Item", "Response"}, label="Hu rx22 trial data")
    prepared = rx22_trials.loc[:, ["Item", "Response"]].copy()
    prepared["Response"] = pd.to_numeric(prepared["Response"], errors="raise")
    if not prepared["Response"].isin([0, 1]).all():
        raise ValueError("Hu rx22 trial responses must be binary 0/1 values")
    counts = (
        prepared.groupby("Item", as_index=False)
        .agg(human_yes=("Response", "sum"), human_total=("Response", "size"))
    )
    counts["human_item_id"] = "rx22_" + counts["Item"].astype(int).astype(str).str.zfill(3)
    if len(counts) != 60:
        raise ValueError(f"Expected 60 Hu rx22 count rows, found {len(counts)}")
    return counts.loc[:, ["human_item_id", "human_yes", "human_total"]]


def _hu_original_inclusion(dataset: pd.Series, scale_id: pd.Series) -> pd.Series:
    included = [
        str(scale).strip().lower()
        not in (
            HU_ORIGINAL_MISSING_STRING_SURPRISAL[str(dataset_name)]
            | HU_ORIGINAL_OTHER_DROPNA_EXCLUSIONS[str(dataset_name)]
        )
        for dataset_name, scale in zip(dataset, scale_id)
    ]
    return pd.Series(included, index=dataset.index, dtype=bool)


def build_hu_canonical_observations(
    human_data: pd.DataFrame,
    rx22_trials: pd.DataFrame,
    no_frame_scores: pd.DataFrame,
    x_frame_scores: pd.DataFrame,
    *,
    source_human_file: str,
    source_rx22_trials_file: str,
    source_no_frame_score_file: str,
    source_x_frame_score_file: str,
) -> pd.DataFrame:
    """Build 309 Hu prompt rows, retaining explicit rate-only count status."""

    human_required = {
        "dataset",
        "item_id",
        "scale_id",
        "template_id",
        "weak_surface",
        "strong_surface",
        "human_si_rate",
        "human_si_rate_source_column",
        "has_hu_test_suite",
    }
    require_columns(human_data, human_required, label="Hu processed human data")
    human = human_data.copy().rename(
        columns={
            "item_id": "human_item_id",
            "weak_surface": "trigger_human",
            "strong_surface": "query_human",
            "human_si_rate": "human_rate",
        }
    )
    if len(human) != 309 or human.duplicated(["dataset", "human_item_id"]).any():
        raise ValueError("Hu processed data must contain 309 unique dataset/item rows")
    human["human_rate"] = pd.to_numeric(human["human_rate"], errors="raise")
    human["hu_original_analysis_included"] = _hu_original_inclusion(
        human["dataset"],
        human["scale_id"],
    )
    included_prompt_rows = (
        human.loc[human["hu_original_analysis_included"]].groupby("dataset").size().to_dict()
    )
    included_scales = (
        human.loc[human["hu_original_analysis_included"]]
        .groupby("dataset")["scale_id"]
        .nunique()
        .to_dict()
    )
    if included_prompt_rows != HU_ORIGINAL_INCLUDED_PROMPT_ROWS:
        raise ValueError(
            "Hu original-analysis prompt coverage changed: "
            f"expected={HU_ORIGINAL_INCLUDED_PROMPT_ROWS}, observed={included_prompt_rows}"
        )
    if included_scales != HU_ORIGINAL_INCLUDED_SCALES:
        raise ValueError(
            "Hu original-analysis scale coverage changed: "
            f"expected={HU_ORIGINAL_INCLUDED_SCALES}, observed={included_scales}"
        )

    no_frame = _prepare_hu_noframe(no_frame_scores)
    x_frame = _prepare_hu_xframe(x_frame_scores)
    provenance = score_provenance(
        no_frame_scores.loc[no_frame_scores["dataset_family"].eq("hu_2023_benchmark")],
        x_frame_scores.loc[x_frame_scores["dataset_family"].eq("hu_2023_benchmark")],
    )

    joined = human.merge(
        no_frame,
        on=["dataset", "human_item_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Some Hu human rows are missing no-frame scores")
    joined = joined.drop(columns="_merge")
    expected_group_id = joined["dataset"].astype(str) + "::" + joined["scale_id"].astype(str)
    if not joined["group_id"].astype(str).eq(expected_group_id).all():
        raise ValueError("Hu no-frame score groups do not match the human scale IDs")
    joined = joined.merge(
        x_frame,
        on=["dataset", "human_item_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Some Hu human rows are missing X-but-not-Y scores")
    joined = joined.drop(columns="_merge")
    if not joined["group_id_x"].astype(str).eq(expected_group_id).all():
        raise ValueError("Hu X-but-not-Y score groups do not match the human scale IDs")

    counts = _rx22_counts(rx22_trials)
    joined = joined.merge(counts, on="human_item_id", how="left", validate="many_to_one")
    exact = joined["dataset"].eq("rx22")
    if joined.loc[exact, ["human_yes", "human_total"]].isna().any(axis=None):
        raise ValueError("Hu rx22 rows are missing recoverable response counts")
    if joined.loc[~exact, ["human_yes", "human_total"]].notna().any(axis=None):
        raise ValueError("Hu rx22 count join unexpectedly matched another dataset")
    recovered_rate = joined.loc[exact, "human_yes"] / joined.loc[exact, "human_total"]
    if not np.allclose(recovered_rate, joined.loc[exact, "human_rate"], atol=1e-12):
        raise ValueError("Recovered Hu rx22 counts do not reproduce the published rates")

    score_source = f"{source_no_frame_score_file};{source_x_frame_score_file}"
    human_source = pd.Series(source_human_file, index=joined.index, dtype=object)
    human_source.loc[exact] = f"{source_human_file};{source_rx22_trials_file}"
    observations = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_family": "hu_2023_benchmark",
            "dataset": joined["dataset"].astype(str),
            "condition": "scalar_inference",
            "item_id": joined["human_item_id"].astype(str),
            "group_id": joined["group_id"].astype(str),
            "context_id": joined["human_item_id"].astype(str),
            "generation_frame": "no_frame",
            "generation_prompt": joined["generation_prompt"].astype(str),
            "trigger": joined["trigger_score"].astype(str),
            "query": joined["query_score"].astype(str),
            "human_outcome": "query_excluded",
            "human_yes": joined["human_yes"],
            "human_total": joined["human_total"],
            "human_rate": joined["human_rate"].astype(float),
            "human_count_status": np.where(exact, "exact", "rate_only"),
            "trigger_logprob_sum": joined["trigger_logprob_sum"].astype(float),
            "query_logprob_sum": joined["query_logprob_sum"].astype(float),
            "trigger_logprob_mean": joined["trigger_logprob_mean"].astype(float),
            "query_logprob_mean": joined["query_logprob_mean"].astype(float),
            "trigger_token_count": joined["trigger_token_count"].astype(int),
            "query_token_count": joined["query_token_count"].astype(int),
            "trigger_tokenization_mode": joined["trigger_tokenization_mode"].astype(str),
            "query_tokenization_mode": joined["query_tokenization_mode"].astype(str),
            "x_but_not_y_applicable": True,
            "x_but_not_y_prompt": joined["x_but_not_y_prompt"].astype(str),
            "x_but_not_y_logprob_sum": joined["x_but_not_y_logprob_sum"].astype(float),
            "x_but_not_y_logprob_mean": joined["x_but_not_y_logprob_mean"].astype(float),
            "x_but_not_y_token_count": joined["x_but_not_y_token_count"].astype(int),
            "x_but_not_y_tokenization_mode": joined[
                "x_but_not_y_tokenization_mode"
            ].astype(str),
            "model_name": provenance.model_name,
            "model_revision": provenance.model_revision,
            "model_provenance_complete": True,
            "source_human_file": human_source,
            "source_score_file": score_source,
            "analysis_inclusion_status": np.select(
                [
                    joined["hu_original_analysis_included"],
                    joined["scale_id"].astype(str).str.lower().isin(
                        HU_ORIGINAL_MISSING_STRING_SURPRISAL["g18"]
                        | HU_ORIGINAL_MISSING_STRING_SURPRISAL["pvt21"]
                        | HU_ORIGINAL_MISSING_STRING_SURPRISAL["rx22"]
                        | HU_ORIGINAL_MISSING_STRING_SURPRISAL["vt16"]
                    ),
                ],
                [
                    "included_hu_original_string_analysis",
                    "excluded_hu_original_missing_gpt2_surprisal",
                ],
                default="excluded_hu_original_source_dropna_missing_lsa",
            ),
            "hu_original_analysis_included": joined[
                "hu_original_analysis_included"
            ].astype(bool),
            "has_hu_test_suite": joined["has_hu_test_suite_x"].astype(bool),
            "human_rate_source_column": joined["human_si_rate_source_column"].astype(str),
            "scale_id": joined["scale_id"].astype(str),
            "template_id": joined["template_id"].astype(str),
            "human_trigger_lemma": joined["trigger_human"].astype(str),
            "human_query_lemma": joined["query_human"].astype(str),
            "x_but_not_y_trigger": joined["trigger_x"].astype(str),
            "x_but_not_y_query": joined["query_x"].astype(str),
        }
    )
    observations = canonical_column_order(
        observations.sort_values(["dataset", "item_id"], ignore_index=True)
    )
    validate_canonical_observations(observations, require_complete_provenance=True)
    return observations


def build_rnx_canonical_observations(
    human_data: pd.DataFrame,
    score_data: pd.DataFrame,
    *,
    source_human_file: str,
    source_score_file: str,
) -> pd.DataFrame:
    """Build the five 60-item Ronai-Xiang condition tables at item-condition grain."""

    human_required = {"experiment", "condition", "item_id", "N", "sum_response", "response_rate"}
    require_columns(human_data, human_required, label="R&X human response rates")
    human = human_data.copy()
    human["human_item_number"] = pd.to_numeric(human["item_id"], errors="raise").astype(int)
    if len(human) != 300 or human.duplicated(["condition", "human_item_number"]).any():
        raise ValueError("R&X human data must contain 300 unique condition/item rows")

    selected = select_score_rows(
        score_data,
        dataset_family="ronai_xiang_2024",
        generation_frame="no_frame",
        expected_rows=300,
        label="R&X no-frame scores",
    )
    provenance = score_provenance(selected)
    score = selected.copy()
    score["human_item_number"] = pd.to_numeric(score["source_row_id"], errors="raise").astype(int)
    score = score.rename(columns={"trigger": "trigger_score", "query": "query_score"})
    joined = human.merge(
        score,
        on=["condition", "human_item_number"],
        how="left",
        validate="one_to_one",
        suffixes=("_human", "_score"),
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("Some R&X human rows are missing no-frame scores")
    joined = joined.drop(columns="_merge")

    observations = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_family": "ronai_xiang_2024",
            "dataset": "ronai_xiang_2024",
            "condition": joined["condition"].astype(str),
            "item_id": joined["item_id_score"].astype(str),
            "group_id": "item::" + joined["group_id"].astype(str),
            "context_id": joined["context_id"].astype(str),
            "generation_frame": "no_frame",
            "generation_prompt": joined["generation_prompt"].astype(str),
            "trigger": joined["trigger_score"].astype(str),
            "query": joined["query_score"].astype(str),
            "human_outcome": "query_excluded",
            "human_yes": pd.to_numeric(joined["sum_response"], errors="raise").astype(int),
            "human_total": pd.to_numeric(joined["N"], errors="raise").astype(int),
            "human_rate": pd.to_numeric(joined["response_rate"], errors="raise"),
            "human_count_status": "exact",
            "trigger_logprob_sum": joined["trigger_logprob_sum"].astype(float),
            "query_logprob_sum": joined["query_logprob_sum"].astype(float),
            "trigger_logprob_mean": joined["trigger_logprob_mean"].astype(float),
            "query_logprob_mean": joined["query_logprob_mean"].astype(float),
            "trigger_token_count": joined["trigger_token_count"].astype(int),
            "query_token_count": joined["query_token_count"].astype(int),
            "trigger_tokenization_mode": joined["trigger_tokenization_mode"].astype(str),
            "query_tokenization_mode": joined["query_tokenization_mode"].astype(str),
            "x_but_not_y_applicable": False,
            "x_but_not_y_prompt": np.nan,
            "x_but_not_y_logprob_sum": np.nan,
            "x_but_not_y_logprob_mean": np.nan,
            "x_but_not_y_token_count": np.nan,
            "x_but_not_y_tokenization_mode": np.nan,
            "model_name": provenance.model_name,
            "model_revision": provenance.model_revision,
            "model_provenance_complete": True,
            "source_human_file": str(source_human_file),
            "source_score_file": str(source_score_file),
            "experiment": joined["experiment"].astype(str),
        }
    )
    observations = canonical_column_order(
        observations.sort_values(["condition", "item_id"], ignore_index=True)
    )
    validate_canonical_observations(
        observations,
        require_complete_provenance=True,
        require_human_counts=True,
    )
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-human", type=Path, default=DEFAULT_FOCUS_HUMAN)
    parser.add_argument("--hu-human", type=Path, default=DEFAULT_HU_HUMAN)
    parser.add_argument("--hu-rx22-trials", type=Path, default=DEFAULT_HU_RX22_TRIALS)
    parser.add_argument("--rnx-human", type=Path, default=DEFAULT_RNX_HUMAN)
    parser.add_argument("--no-frame-scores", type=Path, default=DEFAULT_NO_FRAME_SCORES)
    parser.add_argument("--remaining-scores", type=Path, default=DEFAULT_REMAINING_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_frame_scores = pd.read_csv(args.no_frame_scores)
    remaining_scores = pd.read_csv(args.remaining_scores)

    focus = build_focus_canonical_observations(
        pd.read_csv(args.focus_human),
        remaining_scores,
        source_human_file=_source_path(args.focus_human),
        source_score_file=_source_path(args.remaining_scores),
    )
    hu = build_hu_canonical_observations(
        pd.read_csv(args.hu_human),
        pd.read_csv(args.hu_rx22_trials),
        no_frame_scores,
        remaining_scores,
        source_human_file=_source_path(args.hu_human),
        source_rx22_trials_file=_source_path(args.hu_rx22_trials),
        source_no_frame_score_file=_source_path(args.no_frame_scores),
        source_x_frame_score_file=_source_path(args.remaining_scores),
    )
    rnx = build_rnx_canonical_observations(
        pd.read_csv(args.rnx_human),
        no_frame_scores,
        source_human_file=_source_path(args.rnx_human),
        source_score_file=_source_path(args.no_frame_scores),
    )
    combined = canonical_column_order(pd.concat([focus, hu, rnx], ignore_index=True))
    validate_canonical_observations(combined, require_complete_provenance=True)

    outputs = {
        "novel_focus_observations.csv": focus,
        "hu_2023_observations.csv": hu,
        "ronai_xiang_2024_observations.csv": rnx,
        "all_observations.csv": combined,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, observations in outputs.items():
        output = args.output_dir / filename
        observations.to_csv(output, index=False)
        print(f"[complete] wrote {output}")
        for key, value in asdict(summarize_canonical_observations(observations)).items():
            print(f"  {key}={value}")

    print(
        "[note] Hu g18, pvt21, and vt16 rows are correlation-ready but rate-only; "
        "response-level binomial log likelihood is unavailable without source counts."
    )
    print(
        "[complete] Hu original string-analysis inclusion is frozen at "
        "57/50/67/39 scales for rx22/pvt21/g18/vt16."
    )


if __name__ == "__main__":
    main()
