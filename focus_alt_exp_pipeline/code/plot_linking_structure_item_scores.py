"""Plot item-level model probabilities against human rates for all structures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from build_linking_structure_tables import DATASETS, SAMPLED_COLUMNS, direct_analysis_units
from evaluate_set_variant_grid import item_log_score


MODEL_ORDER = [
    "No linking structure",
    "X but not Y",
    "Set Top-K",
    "Set Top-p",
    "Ordering",
    "Conjunction Top-K",
    "Conjunction Top-p",
    "Disjunction Top-K",
    "Disjunction Top-p",
]


def build_item_table(source_rows: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    expected = oof.loc[oof["variant"].eq("top_k"), [
        "analysis_dataset_id", "analysis_label", "analysis_unit_id", "human_rate"
    ]].drop_duplicates()
    records = []

    direct = expected.merge(
        direct_analysis_units(source_rows),
        on=["analysis_dataset_id", "analysis_unit_id"],
        how="left", suffixes=("_oof", "_direct"), validate="one_to_one",
    )
    if not np.allclose(direct["human_rate_oof"], direct["human_rate_direct"], atol=1e-12):
        raise ValueError("Direct and sampled tables disagree on human rates")
    for model in ("No linking structure", "X but not Y"):
        rows = direct.loc[direct[model].notna()].copy()
        rows["model"] = model
        rows["prediction"] = rows[model]
        rows["human_rate"] = rows["human_rate_oof"]
        records.append(rows[[
            "model", "analysis_dataset_id", "analysis_label", "analysis_unit_id",
            "human_rate", "prediction",
        ]])

    for model, (variant, structure) in SAMPLED_COLUMNS.items():
        rows = oof.loc[oof["variant"].eq(variant), [
            "analysis_dataset_id", "analysis_label", "analysis_unit_id", "human_rate",
            f"{structure}_probability",
        ]].copy()
        rows["model"] = model
        rows = rows.rename(columns={f"{structure}_probability": "prediction"})
        records.append(rows[[
            "model", "analysis_dataset_id", "analysis_label", "analysis_unit_id",
            "human_rate", "prediction",
        ]])

    items = pd.concat(records, ignore_index=True)
    items["item_log_score"] = item_log_score(items["human_rate"], items["prediction"])
    items["exponentiated_log_score"] = np.exp(items["item_log_score"])
    labels = dict(DATASETS)
    items["dataset"] = items["analysis_dataset_id"].map(labels)
    if items["dataset"].isna().any():
        raise ValueError("Unknown dataset in item-level plotting table")
    return items


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def plot_models(
    items: pd.DataFrame,
    output_dir: Path,
    *,
    score_column: str = "item_log_score",
    filename_prefix: str = "item_scores",
    title_metric: str = "proper log scores",
    colorbar_label: str = "Item proper log score (higher is better)",
    linear_unit_scale: bool = False,
) -> None:
    finite_scores = items[score_column].to_numpy(dtype=float)
    if linear_unit_scale:
        norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
        footer = "Dashed line: perfect calibration (p = y). Color uses a fixed linear scale from 0 to 1."
        colorbar_extend = "neither"
    else:
        color_min, color_max = np.nanpercentile(finite_scores, [5, 95])
        norm = Normalize(vmin=float(color_min), vmax=float(color_max), clip=True)
        footer = "Dashed line: perfect calibration (p = y). Colors clipped at the 5th and 95th score percentiles."
        colorbar_extend = "min"
    cmap = plt.get_cmap("RdYlGn")
    dataset_order = [key for key, _ in DATASETS]

    output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        model_rows = items.loc[items["model"].eq(model)]
        fig, axes = plt.subplots(2, 5, figsize=(18, 7.6), sharex=True, sharey=True)
        scatter = None
        for ax, dataset_id in zip(axes.flat, dataset_order):
            rows = model_rows.loc[model_rows["analysis_dataset_id"].eq(dataset_id)]
            ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
            if len(rows):
                scatter = ax.scatter(
                    rows["prediction"], rows["human_rate"],
                    c=rows[score_column], cmap=cmap, norm=norm,
                    s=30, alpha=0.8, edgecolors="black", linewidths=0.25,
                )
                if model == "Disjunction Top-K" and dataset_id == "novel_focus":
                    examples = {
                        "mask::wallet::candy": ("wallet → candy", (10, -18)),
                        "mall::burger joint::pretzel stand": (
                            "burger joint → pretzel stand", (10, 10)
                        ),
                    }
                    for unit_id, (label, offset) in examples.items():
                        match = rows.loc[rows["analysis_unit_id"].eq(unit_id)]
                        if len(match):
                            point = match.iloc[0]
                            ax.annotate(
                                label,
                                (point["prediction"], point["human_rate"]),
                                xytext=offset, textcoords="offset points",
                                fontsize=7, arrowprops={"arrowstyle": "-", "lw": 0.7},
                            )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, color="#555555")
                ax.set_facecolor("#E5E5E5")
            ax.set_title(dict(DATASETS)[dataset_id], fontsize=10)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.15)
        for ax in axes[1, :]:
            ax.set_xlabel("Model probability (p)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Human exclusion rate (y)")
        fig.suptitle(
            f"{model}: item-level predictions and {title_metric}",
            fontsize=16, y=0.995,
        )
        fig.subplots_adjust(
            left=0.06, right=0.90, bottom=0.09, top=0.91,
            wspace=0.20, hspace=0.28,
        )
        if scatter is not None:
            colorbar_ax = fig.add_axes([0.925, 0.18, 0.012, 0.64])
            colorbar = fig.colorbar(
                scatter, cax=colorbar_ax, extend=colorbar_extend
            )
            colorbar.set_label(colorbar_label)
        fig.text(
            0.5, 0.012, footer,
            ha="center", fontsize=9,
        )
        fig.savefig(output_dir / f"{filename_prefix}_{_slug(model)}.png", dpi=180)
        plt.close(fig)


def plot_novel_contexts(
    items: pd.DataFrame,
    source_rows: pd.DataFrame,
    output_dir: Path,
    *,
    score_column: str = "item_log_score",
    title_metric: str = "item-level predictions",
    colorbar_label: str = "Item proper log score (higher is better)",
    linear_unit_scale: bool = False,
) -> pd.DataFrame:
    mapping = source_rows.loc[
        source_rows["dataset_family"].eq("novel_focus"),
        ["item_id", "context_id", "trigger", "query"],
    ].drop_duplicates()
    if mapping.duplicated("item_id").any():
        raise ValueError("Novel Focus item IDs do not map uniquely to contexts")
    novel = items.loc[items["analysis_dataset_id"].eq("novel_focus")].merge(
        mapping,
        left_on="analysis_unit_id", right_on="item_id",
        how="left", validate="many_to_one",
    )
    if novel["context_id"].isna().any():
        raise ValueError("Novel Focus plotting rows are missing context IDs")

    finite_scores = novel[score_column].to_numpy(dtype=float)
    if linear_unit_scale:
        norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
        footer = "Dashed line: perfect calibration (p = y). Color uses a fixed linear scale from 0 to 1."
        colorbar_extend = "neither"
    else:
        color_min, color_max = np.nanpercentile(finite_scores, [5, 95])
        norm = Normalize(vmin=float(color_min), vmax=float(color_max), clip=True)
        footer = "Dashed line: perfect calibration (p = y). Colors share one Novel Focus scale and are clipped at the 5th and 95th percentiles."
        colorbar_extend = "min"
    cmap = plt.get_cmap("RdYlGn")
    output_dir.mkdir(parents=True, exist_ok=True)

    for context_id, context_rows in novel.groupby("context_id", sort=True):
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5), sharex=True, sharey=True)
        scatter = None
        for ax, model in zip(axes.flat, MODEL_ORDER):
            rows = context_rows.loc[context_rows["model"].eq(model)]
            ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
            scatter = ax.scatter(
                rows["prediction"], rows["human_rate"],
                c=rows[score_column], cmap=cmap, norm=norm,
                s=42, alpha=0.82, edgecolors="black", linewidths=0.3,
            )
            ax.set_title(f"{model} (N={len(rows)})", fontsize=10)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.15)
        for ax in axes[2, :]:
            ax.set_xlabel("Model probability (p)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Human exclusion rate (y)")
        fig.suptitle(
            f"Novel Focus context: {context_id} — {title_metric}",
            fontsize=16, y=0.985,
        )
        fig.subplots_adjust(
            left=0.07, right=0.90, bottom=0.07, top=0.93,
            wspace=0.18, hspace=0.22,
        )
        colorbar_ax = fig.add_axes([0.925, 0.18, 0.015, 0.64])
        colorbar = fig.colorbar(scatter, cax=colorbar_ax, extend=colorbar_extend)
        colorbar.set_label(colorbar_label)
        fig.text(
            0.5, 0.018, footer,
            ha="center", fontsize=9,
        )
        fig.savefig(output_dir / f"novel_focus_context_{_slug(str(context_id))}.png", dpi=180)
        plt.close(fig)
    return novel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = pd.read_csv(args.source_rows)
    items = build_item_table(
        source_rows,
        pd.read_csv(args.results_dir / "cv_results" / "oof_predictions.csv"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(args.output_dir / "item_level_predictions_and_log_scores.csv", index=False)
    plot_models(items, args.output_dir)
    novel = plot_novel_contexts(
        items, source_rows, args.output_dir / "novel_focus_by_context"
    )
    novel.to_csv(
        args.output_dir / "novel_focus_by_context" / "novel_focus_item_level_predictions.csv",
        index=False,
    )
    exponentiated_dir = args.output_dir.parent / "item_level_scatterplots_exponentiated"
    plot_models(
        items, exponentiated_dir,
        score_column="exponentiated_log_score",
        filename_prefix="item_exponentiated_log_score",
        title_metric="exponentiated log scores",
        colorbar_label="Exponentiated item log score",
        linear_unit_scale=True,
    )
    exponentiated_novel = plot_novel_contexts(
        items, source_rows, exponentiated_dir / "novel_focus_by_context",
        score_column="exponentiated_log_score",
        title_metric="exponentiated item log scores",
        colorbar_label="Exponentiated item log score",
        linear_unit_scale=True,
    )
    exponentiated_novel.to_csv(
        exponentiated_dir / "novel_focus_by_context" / "novel_focus_item_level_predictions.csv",
        index=False,
    )
    print(f"[complete] rows={len(items)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
