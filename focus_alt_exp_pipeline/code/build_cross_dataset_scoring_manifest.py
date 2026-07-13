"""Build the Hu and Ronai-Xiang no-frame Qwen scoring manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .scoring_manifest import (
        MANIFEST_COLUMNS,
        MANIFEST_VERSION,
        summarize_scoring_manifest,
        validate_scoring_manifest,
    )
except ImportError:
    from scoring_manifest import (
        MANIFEST_COLUMNS,
        MANIFEST_VERSION,
        summarize_scoring_manifest,
        validate_scoring_manifest,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EXPERIMENT_ROOT = WORKSPACE_ROOT / "experiment_ronai&xiang"

DEFAULT_HU_PROMPTS = (
    EXPERIMENT_ROOT
    / "jen_hu_modeling"
    / "stimuli"
    / "qwen_no_frame_strong_prompts.csv"
)
DEFAULT_RNX_1 = EXPERIMENT_ROOT / "stimuli_prompts" / "experiment_1_scoring_stimuli.csv"
DEFAULT_RNX_2 = EXPERIMENT_ROOT / "stimuli_prompts" / "experiment_2_scoring_stimuli.csv"
DEFAULT_RNX_3 = EXPERIMENT_ROOT / "stimuli_prompts" / "experiment_3_scoring_stimuli.csv"
DEFAULT_RNX_4 = EXPERIMENT_ROOT / "stimuli_prompts" / "experiment_4_scoring_stimuli.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "focus_alt_exp_pipeline"
    / "scoring_manifests"
    / "hu_rnx_no_frame_manifest.csv"
)


def _source_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_columns(df: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def build_hu_manifest(hu_prompts: pd.DataFrame, *, source_file: str) -> pd.DataFrame:
    required = {
        "dataset",
        "item_id",
        "scale_id",
        "template_id",
        "weak_surface_in_sentence",
        "target_text",
        "context_text",
        "moved_prompt_boundary_text",
        "has_hu_test_suite",
        "prompt_source",
    }
    _require_columns(hu_prompts, required, label="Hu no-frame prompts")
    prepared = hu_prompts.copy()
    prompt = prepared["context_text"].fillna("").astype(str) + prepared[
        "moved_prompt_boundary_text"
    ].fillna("").astype(str)
    suite_available = prepared["has_hu_test_suite"].fillna(False).astype(bool)

    manifest = pd.DataFrame(
        {
            "manifest_version": MANIFEST_VERSION,
            "dataset_family": "hu_2023_benchmark",
            "dataset": prepared["dataset"].astype(str),
            "condition": "scalar_inference",
            "item_id": prepared["item_id"].astype(str),
            "group_id": prepared["dataset"].astype(str)
            + "::"
            + prepared["scale_id"].astype(str),
            "context_id": prepared["item_id"].astype(str),
            "generation_frame": "no_frame",
            "generation_prompt": prompt,
            "trigger": prepared["weak_surface_in_sentence"].astype(str).str.strip(),
            "query": prepared["target_text"].astype(str).str.strip(),
            "analysis_inclusion_status": np.where(
                suite_available,
                "pending_hu_exact_filter_suite_available",
                "pending_hu_exact_filter_no_suite",
            ),
            "has_hu_test_suite": suite_available,
            "source_prompt_file": source_file,
            "source_row_id": prepared["item_id"].astype(str),
            "prompt_provenance": prepared["prompt_source"].astype(str),
        }
    )
    if len(manifest) != 309:
        raise ValueError(f"Expected 309 Hu context rows, found {len(manifest)}")
    return manifest


def _rnx_base_columns(
    rows: pd.DataFrame,
    *,
    condition: str,
    prompt: pd.Series,
    source_file: str,
    prompt_provenance: str,
) -> pd.DataFrame:
    required = {"item_id", "weaker_surface", "stronger_surface_guess"}
    _require_columns(rows, required, label=f"R&X {condition} rows")
    item_ids = rows["item_id"].astype(int)
    return pd.DataFrame(
        {
            "manifest_version": MANIFEST_VERSION,
            "dataset_family": "ronai_xiang_2024",
            "dataset": "ronai_xiang_2024",
            "condition": condition,
            "item_id": condition + "::" + item_ids.astype(str).str.zfill(2),
            "group_id": item_ids.astype(str).str.zfill(2),
            "context_id": condition + "::" + item_ids.astype(str).str.zfill(2),
            "generation_frame": "no_frame",
            "generation_prompt": prompt.astype(str),
            "trigger": rows["weaker_surface"].astype(str).str.strip(),
            "query": rows["stronger_surface_guess"].astype(str).str.strip(),
            "analysis_inclusion_status": "included",
            "has_hu_test_suite": np.nan,
            "source_prompt_file": source_file,
            "source_row_id": item_ids.astype(str),
            "prompt_provenance": prompt_provenance,
        }
    )


def build_rnx_manifest(
    experiment_1: pd.DataFrame,
    experiment_2: pd.DataFrame,
    experiment_3: pd.DataFrame,
    experiment_4: pd.DataFrame,
    *,
    source_files: dict[str, str],
) -> pd.DataFrame:
    _require_columns(experiment_1, {"item_id", "prompt_prefix"}, label="R&X Experiment 1")
    _require_columns(
        experiment_2,
        {"item_id", "condition", "prompt_prefix_for_qwen"},
        label="R&X Experiment 2",
    )
    _require_columns(experiment_3, {"item_id", "condition"}, label="R&X Experiment 3")
    _require_columns(experiment_4, {"item_id", "condition"}, label="R&X Experiment 4")

    exp1 = experiment_1.sort_values("item_id", ignore_index=True)
    exp2 = experiment_2.sort_values(["condition", "item_id"], ignore_index=True)
    exp3 = experiment_3.sort_values("item_id", ignore_index=True)
    exp4 = experiment_4.sort_values("item_id", ignore_index=True)
    weak_qud = exp2.loc[exp2["condition"].eq("Eweak")].copy()
    strong_qud = exp2.loc[exp2["condition"].eq("Estrong")].copy()

    for label, rows, expected in [
        ("ESI", exp1, 60),
        ("Eweak", weak_qud, 60),
        ("Estrong", strong_qud, 60),
        ("Eonly", exp3, 60),
        ("Eonlystrong", exp4, 60),
    ]:
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} R&X {label} rows, found {len(rows)}")

    exp1_by_item = exp1.set_index("item_id")
    strong_by_item = strong_qud.set_index("item_id")
    exp3_prompt = exp3["item_id"].map(exp1_by_item["prompt_prefix"])
    exp4_prompt = exp4["item_id"].map(strong_by_item["prompt_prefix_for_qwen"])
    if exp3_prompt.isna().any() or exp4_prompt.isna().any():
        raise ValueError("Could not recover matched no-frame prompts for R&X only conditions")

    manifests = [
        _rnx_base_columns(
            exp1,
            condition="ESI",
            prompt=exp1["prompt_prefix"],
            source_file=source_files["experiment_1"],
            prompt_provenance="experiment_1_plain_answer",
        ),
        _rnx_base_columns(
            weak_qud,
            condition="Eweak",
            prompt=weak_qud["prompt_prefix_for_qwen"],
            source_file=source_files["experiment_2"],
            prompt_provenance="experiment_2_weak_qud_neutral_answer",
        ),
        _rnx_base_columns(
            strong_qud,
            condition="Estrong",
            prompt=strong_qud["prompt_prefix_for_qwen"],
            source_file=source_files["experiment_2"],
            prompt_provenance="experiment_2_strong_qud_neutral_answer",
        ),
        _rnx_base_columns(
            exp3,
            condition="Eonly",
            prompt=exp3_prompt,
            source_file=source_files["experiment_3"],
            prompt_provenance="only_removed_by_reusing_experiment_1_prompt",
        ),
        _rnx_base_columns(
            exp4,
            condition="Eonlystrong",
            prompt=exp4_prompt,
            source_file=source_files["experiment_4"],
            prompt_provenance="only_removed_by_reusing_experiment_2_Estrong_prompt",
        ),
    ]
    manifest = pd.concat(manifests, ignore_index=True)
    if len(manifest) != 300:
        raise ValueError(f"Expected 300 R&X item-condition rows, found {len(manifest)}")
    return manifest


def build_cross_dataset_manifest(
    hu_prompts: pd.DataFrame,
    experiment_1: pd.DataFrame,
    experiment_2: pd.DataFrame,
    experiment_3: pd.DataFrame,
    experiment_4: pd.DataFrame,
    *,
    source_files: dict[str, str],
) -> pd.DataFrame:
    hu = build_hu_manifest(hu_prompts, source_file=source_files["hu"])
    rnx = build_rnx_manifest(
        experiment_1,
        experiment_2,
        experiment_3,
        experiment_4,
        source_files=source_files,
    )
    manifest = pd.concat([hu, rnx], ignore_index=True)
    manifest = manifest.loc[:, MANIFEST_COLUMNS].sort_values(
        ["dataset_family", "dataset", "condition", "item_id"],
        ignore_index=True,
    )
    validate_scoring_manifest(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Hu/R&X no-frame Qwen manifest.")
    parser.add_argument("--hu-prompts", type=Path, default=DEFAULT_HU_PROMPTS)
    parser.add_argument("--rnx-experiment-1", type=Path, default=DEFAULT_RNX_1)
    parser.add_argument("--rnx-experiment-2", type=Path, default=DEFAULT_RNX_2)
    parser.add_argument("--rnx-experiment-3", type=Path, default=DEFAULT_RNX_3)
    parser.add_argument("--rnx-experiment-4", type=Path, default=DEFAULT_RNX_4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "hu": args.hu_prompts,
        "experiment_1": args.rnx_experiment_1,
        "experiment_2": args.rnx_experiment_2,
        "experiment_3": args.rnx_experiment_3,
        "experiment_4": args.rnx_experiment_4,
    }
    manifest = build_cross_dataset_manifest(
        pd.read_csv(paths["hu"]),
        pd.read_csv(paths["experiment_1"]),
        pd.read_csv(paths["experiment_2"]),
        pd.read_csv(paths["experiment_3"]),
        pd.read_csv(paths["experiment_4"]),
        source_files={key: _source_label(path) for key, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    summary = asdict(summarize_scoring_manifest(manifest))
    print(f"[complete] wrote {args.output}")
    for key, value in summary.items():
        print(f"  {key}={value}")
    print(
        "[note] All Hu rows are scored; the published-analysis inclusion rule "
        "is applied later by the canonical-data builder."
    )


if __name__ == "__main__":
    main()
