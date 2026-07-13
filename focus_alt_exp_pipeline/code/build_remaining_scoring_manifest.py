"""Build the remaining focus and Hu Qwen scoring manifest.

This manifest covers:

* novel-focus no-frame trigger and query scores;
* novel-focus query scores after the X-but-not-Y frame; and
* Hu query scores after the X-but-not-Y frame.

R&X does not receive X-but-not-Y rows, and alternative-structure models use
only the no-frame rows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .scoring_manifest import (
        FRAME_AWARE_MANIFEST_VERSION,
        MANIFEST_COLUMNS,
        SCORING_CONTROL_COLUMNS,
        summarize_scoring_manifest,
        validate_scoring_manifest,
    )
except ImportError:
    from scoring_manifest import (
        FRAME_AWARE_MANIFEST_VERSION,
        MANIFEST_COLUMNS,
        SCORING_CONTROL_COLUMNS,
        summarize_scoring_manifest,
        validate_scoring_manifest,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EXPERIMENT_ROOT = WORKSPACE_ROOT / "experiment_ronai&xiang"

DEFAULT_FOCUS_HUMAN = (
    REPO_ROOT / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
)
DEFAULT_FOCUS_PROMPTS = REPO_ROOT / "prompts" / "prompt_files" / "prompts_llm_next_word.csv"
DEFAULT_HU_X_PROMPTS = (
    EXPERIMENT_ROOT
    / "jen_hu_modeling"
    / "stimuli"
    / "qwen_exact_strong_prompts.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "focus_alt_exp_pipeline"
    / "scoring_manifests"
    / "focus_hu_remaining_qwen_manifest.csv"
)
OUTPUT_COLUMNS = (*MANIFEST_COLUMNS, *SCORING_CONTROL_COLUMNS)


def _source_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_columns(df: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _clean_candidate(value: object) -> str:
    candidate = str(value).strip().lower()
    if candidate.startswith("a "):
        return candidate[2:]
    if candidate.startswith("an "):
        return candidate[3:]
    return candidate


def _focus_pairs(human_data: pd.DataFrame) -> pd.DataFrame:
    required = {"story", "cleaned_trigger", "cleaned_query"}
    _require_columns(human_data, required, label="Focus human data")
    pairs = human_data.loc[:, sorted(required)].dropna().copy()
    pairs = pairs.rename(
        columns={"cleaned_trigger": "trigger", "cleaned_query": "query"}
    )
    pairs["story"] = pairs["story"].astype(str).str.strip()
    pairs["trigger"] = pairs["trigger"].map(_clean_candidate)
    pairs["query"] = pairs["query"].map(_clean_candidate)
    pairs = pairs.loc[
        pairs["story"].ne("")
        & pairs["trigger"].ne("")
        & pairs["query"].ne("")
        & pairs["trigger"].ne(pairs["query"])
    ].drop_duplicates(ignore_index=True)
    pairs = pairs.sort_values(["story", "trigger", "query"], ignore_index=True)

    if len(pairs) != 480:
        raise ValueError(f"Expected 480 unique focus trigger/query pairs, found {len(pairs)}")
    if pairs["story"].nunique() != 16:
        raise ValueError(f"Expected 16 focus contexts, found {pairs['story'].nunique()}")
    per_story = pairs.groupby("story").size()
    if not per_story.eq(30).all():
        raise ValueError(
            "Each focus context must contain 30 ordered trigger/query pairs; "
            f"observed={per_story.to_dict()}"
        )
    return pairs


def build_focus_manifest(
    human_data: pd.DataFrame,
    focus_prompts: pd.DataFrame,
    *,
    human_source_file: str,
    prompt_source_file: str,
) -> pd.DataFrame:
    _require_columns(focus_prompts, {"story", "prompt"}, label="Focus prompts")
    prompt_rows = focus_prompts[["story", "prompt"]].drop_duplicates().copy()
    prompt_rows["story"] = prompt_rows["story"].astype(str).str.strip()
    if prompt_rows["story"].duplicated().any():
        raise ValueError("Focus prompt table has multiple prompts for one context")
    prompt_map = dict(zip(prompt_rows["story"], prompt_rows["prompt"]))

    pairs = _focus_pairs(human_data)
    missing_prompts = sorted(set(pairs["story"]).difference(prompt_map))
    if missing_prompts:
        raise ValueError(f"Focus contexts are missing prompts: {missing_prompts}")

    records: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        base_prompt = str(prompt_map[row.story]).rstrip() + " "
        observation_id = f"{row.story}::{row.trigger}::{row.query}"
        shared = {
            "manifest_version": FRAME_AWARE_MANIFEST_VERSION,
            "dataset_family": "novel_focus",
            "dataset": "focus_alternative_study",
            "condition": "focus_only",
            "group_id": row.story,
            "context_id": row.story,
            "trigger": row.trigger,
            "query": row.query,
            "analysis_inclusion_status": "included",
            "has_hu_test_suite": np.nan,
            "source_prompt_file": prompt_source_file,
            "source_row_id": observation_id,
        }
        records.append(
            {
                **shared,
                "item_id": f"{observation_id}::no_frame",
                "generation_frame": "no_frame",
                "generation_prompt": base_prompt,
                "prompt_provenance": (
                    "focus_story_prompt_with_cleaned_trigger_and_query_from_"
                    f"{human_source_file}"
                ),
                "score_trigger": True,
                "score_query": True,
            }
        )
        records.append(
            {
                **shared,
                "item_id": f"{observation_id}::x_but_not_y",
                "generation_frame": "x_but_not_y",
                "generation_prompt": f"{base_prompt.rstrip()} {row.trigger} but not ",
                "prompt_provenance": (
                    "focus_story_prompt_plus_cleaned_trigger_but_not;query_only"
                ),
                "score_trigger": False,
                "score_query": True,
            }
        )

    manifest = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if len(manifest) != 960:
        raise ValueError(f"Expected 960 focus scoring rows, found {len(manifest)}")
    return manifest


def build_hu_x_but_not_y_manifest(
    hu_prompts: pd.DataFrame,
    *,
    source_file: str,
) -> pd.DataFrame:
    required = {
        "prompt_id",
        "dataset",
        "item_id",
        "scale_id",
        "weak_surface",
        "target_text",
        "context_text",
        "has_hu_test_suite",
        "prompt_source",
    }
    _require_columns(hu_prompts, required, label="Hu X-but-not-Y prompts")
    rows = hu_prompts.copy()
    suite_available = rows["has_hu_test_suite"].fillna(False).astype(bool)
    manifest = pd.DataFrame(
        {
            "manifest_version": FRAME_AWARE_MANIFEST_VERSION,
            "dataset_family": "hu_2023_benchmark",
            "dataset": rows["dataset"].astype(str),
            "condition": "scalar_inference",
            "item_id": rows["prompt_id"].astype(str) + "::x_but_not_y",
            "group_id": rows["dataset"].astype(str)
            + "::"
            + rows["scale_id"].astype(str),
            "context_id": rows["item_id"].astype(str),
            "generation_frame": "x_but_not_y",
            "generation_prompt": rows["context_text"].astype(str).str.rstrip() + " ",
            "trigger": rows["weak_surface"].astype(str).str.strip(),
            "query": rows["target_text"].astype(str).str.strip(),
            "analysis_inclusion_status": np.where(
                suite_available,
                "pending_hu_exact_filter_suite_available",
                "pending_hu_exact_filter_no_suite",
            ),
            "has_hu_test_suite": suite_available,
            "source_prompt_file": source_file,
            "source_row_id": rows["prompt_id"].astype(str),
            "prompt_provenance": rows["prompt_source"].astype(str)
            + ";query_only_standardized_rescore",
            "score_trigger": False,
            "score_query": True,
        }
    )
    if len(manifest) != 309:
        raise ValueError(f"Expected 309 Hu X-but-not-Y rows, found {len(manifest)}")
    return manifest.loc[:, OUTPUT_COLUMNS]


def build_remaining_manifest(
    focus_human: pd.DataFrame,
    focus_prompts: pd.DataFrame,
    hu_x_prompts: pd.DataFrame,
    *,
    source_files: dict[str, str],
) -> pd.DataFrame:
    focus = build_focus_manifest(
        focus_human,
        focus_prompts,
        human_source_file=source_files["focus_human"],
        prompt_source_file=source_files["focus_prompts"],
    )
    hu = build_hu_x_but_not_y_manifest(
        hu_x_prompts,
        source_file=source_files["hu_x_prompts"],
    )
    manifest = pd.concat([focus, hu], ignore_index=True)
    manifest = manifest.sort_values(
        ["dataset_family", "dataset", "generation_frame", "item_id"],
        ignore_index=True,
    )
    validate_scoring_manifest(manifest)
    if len(manifest) != 1269:
        raise ValueError(f"Expected 1269 remaining scoring rows, found {len(manifest)}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build focus and Hu remaining-frame Qwen scoring manifest."
    )
    parser.add_argument("--focus-human", type=Path, default=DEFAULT_FOCUS_HUMAN)
    parser.add_argument("--focus-prompts", type=Path, default=DEFAULT_FOCUS_PROMPTS)
    parser.add_argument("--hu-x-prompts", type=Path, default=DEFAULT_HU_X_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "focus_human": args.focus_human,
        "focus_prompts": args.focus_prompts,
        "hu_x_prompts": args.hu_x_prompts,
    }
    manifest = build_remaining_manifest(
        pd.read_csv(paths["focus_human"]),
        pd.read_csv(paths["focus_prompts"]),
        pd.read_csv(paths["hu_x_prompts"]),
        source_files={key: _source_label(path) for key, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"[complete] wrote {args.output}")
    for key, value in asdict(summarize_scoring_manifest(manifest)).items():
        print(f"  {key}={value}")
    frame_counts = manifest.groupby(["dataset_family", "generation_frame"]).size()
    print(frame_counts.to_string())
    print(
        "[note] All Hu rows are scored; the published-analysis inclusion rule "
        "is applied later by the canonical-data builder."
    )


if __name__ == "__main__":
    main()
