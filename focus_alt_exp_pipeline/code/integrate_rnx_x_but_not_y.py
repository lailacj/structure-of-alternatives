"""After cluster scoring, add R&X framed predictions to a new corrected results bundle.

No inference, sampling, boundary selection, or GPU work. Parent files are read-only.
Only framed source fields and the source-hash lineage of copied predictions change.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from canonical_observations import validate_canonical_observations
from rnx_contrastive import (
    CONDITIONS, DEFAULT_MANIFEST, DEFAULT_PARENT_VIEWER, DEFAULT_SCORES, DEFAULT_SOURCE, FAMILY, FRAME_COLUMNS,
    PIPELINE, attach_scores, sha256, validate_scores,
)


def write_source_overlay(parent: Path, output: Path, attached: pd.DataFrame) -> None:
    """Keep original serialized neutral values; change only R&X framed fields."""
    with parent.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    fields += [name for name in FRAME_COLUMNS if name not in fields]
    for index, row in enumerate(rows):
        if row["dataset_family"] == FAMILY:
            for name in FRAME_COLUMNS:
                value = attached.iloc[index][name]
                row[name] = "" if pd.isna(value) else str(value)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_predictions(parent: Path, output: Path, *, parent_hash: str, new_hash: str) -> dict:
    """Update only the source hash after proving sampled inputs were preserved."""
    with parent.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    if not rows or "source_rows_sha256" not in fields or any(row["source_rows_sha256"] != parent_hash for row in rows):
        raise ValueError(f"Parent prediction hash disagrees with corrected source: {parent}")
    for row in rows:
        row["source_rows_sha256"] = new_hash
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"parent_sha256": sha256(parent), "sha256": sha256(output), "rows": len(rows),
            "changed_columns": ["source_rows_sha256"], "sampled_values_and_folds": "copied unchanged"}


def condition_changes(source: pd.DataFrame, output: Path) -> None:
    rows = source.loc[source.dataset_family.eq(FAMILY)].copy()
    rows["model_probability"] = np.exp(rows.x_but_not_y_logprob_sum)
    records, summaries = [], []
    for left, right in [("ESI", "Eweak"), ("Eweak", "Estrong"),
                        ("ESI", "Eonly"), ("Estrong", "Eonlystrong")]:
        a = rows.loc[rows.condition.eq(left)].set_index("group_id")
        b = rows.loc[rows.condition.eq(right)].set_index("group_id")
        if len(a) != 60 or set(a.index) != set(b.index):
            raise ValueError("Paired R&X condition items do not match")
        b = b.loc[a.index]
        model_delta = b.model_probability - a.model_probability
        if "only" in right and not model_delta.eq(0).all():
            raise ValueError("Matched only conditions must have identical framed predictions")
        for item in a.index:
            records.append(dict(contrast=f"{right} minus {left}", group_id=item,
                                human_before=a.at[item, "human_rate"], human_after=b.at[item, "human_rate"],
                                human_delta=b.at[item, "human_rate"]-a.at[item, "human_rate"],
                                model_before=a.at[item, "model_probability"], model_after=b.at[item, "model_probability"],
                                model_delta=model_delta.loc[item],
                                model_logprob_delta=b.at[item, "x_but_not_y_logprob_sum"]-a.at[item, "x_but_not_y_logprob_sum"]))
        summaries.append(dict(contrast=f"{right} minus {left}", n=60,
                              mean_human_delta=float((b.human_rate-a.human_rate).mean()),
                              mean_model_delta=float(model_delta.mean()),
                              mean_model_logprob_delta=float((b.x_but_not_y_logprob_sum-a.x_but_not_y_logprob_sum).mean()),
                              all_model_predictions_identical=bool(model_delta.eq(0).all())))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "paired_condition_changes.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "condition_change_summary.csv", index=False)


def integrate(*, parent_results: Path, manifest_path: Path, scores_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; refusing overwrite: {output}")
    if parent_results.resolve() in output.resolve().parents or output.resolve() in parent_results.resolve().parents:
        raise ValueError("Output and parent results must be separate bundles")
    source_path = parent_results / "inputs/source_rows.csv"
    source = pd.read_csv(source_path)
    if "neutral_score_array_sha256" not in source or source.neutral_score_array_sha256.isna().any():
        raise ValueError("Integration requires a complete corrected parent bundle")
    rnx = source.dataset_family.eq(FAMILY)
    if source.loc[rnx, "x_but_not_y_logprob_sum"].notna().any():
        raise ValueError("Parent already contains R&X framed scores; choose the original corrected bundle")
    manifest, scores = pd.read_csv(manifest_path), pd.read_csv(scores_path)
    validate_scores(scores, manifest, manifest_sha256=sha256(manifest_path))
    attached = attach_scores(source, scores, artifact="inputs/rnx_x_but_not_y_qwen_scores.csv")
    validate_canonical_observations(attached, require_complete_provenance=True)
    parent_hash = sha256(source_path)
    # Freeze a verified, self-contained extension; all writes happen in a temporary sibling.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rnx-contrastive-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        inputs, cv = stage / "inputs", stage / "cv_results"
        inputs.mkdir(parents=True)
        cv.mkdir()
        for path in (parent_results / "inputs").iterdir():
            if path.is_file() and path.name != "source_rows.csv":
                name = "parent_" + path.name if path.name in ("score_artifact_provenance.json", "framed_scoring_audit.json") else path.name
                shutil.copy2(path, inputs / name)
        write_source_overlay(source_path, inputs / "source_rows.csv", attached)
        new_hash = sha256(inputs / "source_rows.csv")
        prediction_audit = {}
        for relative in ("prediction_grid.csv", "cv_results/oof_predictions.csv"):
            prediction_audit[relative] = copy_predictions(parent_results / relative, stage / relative,
                                                         parent_hash=parent_hash, new_hash=new_hash)
        for path in (parent_results / "cv_results").iterdir():
            if path.is_file() and path.name != "oof_predictions.csv":
                shutil.copy2(path, cv / path.name)
        shutil.copy2(manifest_path, inputs / "rnx_x_but_not_y_manifest.csv")
        shutil.copy2(scores_path, inputs / "rnx_x_but_not_y_qwen_scores.csv")
        attached.to_csv(inputs / "all_observations.csv", index=False)
        attached.loc[rnx].to_csv(inputs / "ronai_xiang_2024_observations.csv", index=False)
        condition_changes(attached, stage / "rnx_contrastive")
        report = {"parent_results": str(parent_results.resolve()), "parent_source_rows_sha256": parent_hash,
                  "returned_scores": str(scores_path.resolve()),
                  "source_rows_sha256": new_hash, "framed_manifest_sha256": sha256(manifest_path),
                  "framed_scores_sha256": sha256(scores_path), "rnx_framed_rows": 300,
                  "unique_prompt_target_pairs": len(manifest[["generation_prompt", "query"]].drop_duplicates()),
                  "changed_source_fields": FRAME_COLUMNS,
                  "neutral_scores_humans_sampling_inputs": "verified unchanged before extending source lineage",
                  "copied_predictions": prediction_audit, "fold_selection": "copied unchanged; no refitting",
                  "parent_audits": "parent_* audit files describe the original neutral and pre-existing framed inputs",
                  "scoring_boundary": "single-space-exact-concat-v1",
                  "prompt_policy": "weaker X; score stronger Y; only omitted; QUD retained"}
        (inputs / "rnx_contrastive_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        (stage / "README.md").write_text(
            "# Corrected results with R&X X-but-not-Y\n\n"
            "This bundle adds 300 query-only contrastive predictions to the verified corrected parent. "
            "Neutral scores, sampled predictions, folds and selected boundaries are reused unchanged. "
            "Only framed source fields and copied prediction source-hash references change. "
            "See inputs/rnx_contrastive_provenance.json for parent hashes and the extension audit.\n\n"
            "Only is omitted: ESI/Eonly and Estrong/Eonlystrong predictions are identical. "
            "rnx_contrastive/ contains descriptive matched-item condition changes.\n")
        os.replace(stage, output)
    print(f"[complete] added R&X framed predictions in {output}; no model loaded or sampling performed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-results", type=Path, default=DEFAULT_SOURCE.parent.parent)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-reports", action="store_true", help="Also regenerate aligned tables and a separate offline viewer (CPU only)")
    parser.add_argument("--parent-viewer", type=Path, default=DEFAULT_PARENT_VIEWER,
                        help="Verified parent viewer whose saved vocabulary distributions can be reused")
    parser.add_argument("--log-probs-dir", type=Path, help="Instead of parent-viewer reuse, read existing corrected neutral arrays")
    args = parser.parse_args()
    if args.build_reports and not args.log_probs_dir and not args.parent_viewer.is_file():
        parser.error("Report building needs --parent-viewer with the saved corrected distributions or --log-probs-dir; scoring is not required")
    integrate(parent_results=args.parent_results, manifest_path=args.manifest, scores_path=args.scores, output=args.output_dir)
    if args.build_reports:
        subprocess.run([sys.executable, str(PIPELINE / "code/build_linking_structure_tables.py"),
                        "--source-rows", str(args.output_dir / "inputs/source_rows.csv"),
                        "--results-dir", str(args.output_dir), "--output-dir", str(args.output_dir / "linking_structure_tables")], check=True)
        distribution_args = (["--log-probs-dir", str(args.log_probs_dir)] if args.log_probs_dir
                             else ["--reuse-distributions-from", str(args.parent_viewer)])
        subprocess.run([sys.executable, str(PIPELINE / "code/build_results_viewer.py"),
                        "--results-dir", str(args.output_dir), "--output", str(args.output_dir / "results_viewer.html"),
                        *distribution_args], check=True)


if __name__ == "__main__":
    main()
