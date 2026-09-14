"""Build advisor-ready tables, figures, and notes from set-variant OOF results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_set_variant_grid import item_log_score


STRUCTURES = ("set", "ordering", "conjunction", "disjunction")
LINKING_SHORT_LABELS = [
    "No linking",
    "X but not Y",
    "Set Top-K",
    "Set Top-p",
    "Ordering",
    "Conj. Top-K",
    "Conj. Top-p",
    "Disj. Top-K",
    "Disj. Top-p",
]


def _dataset_base_rate(oof: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    units = oof.loc[oof["variant"].eq("top_k"), [
        "analysis_dataset_id", "analysis_label", "analysis_unit_id", "cv_fold", "human_rate"
    ]].drop_duplicates()
    records = []
    for (dataset_id, label), rows in units.groupby(["analysis_dataset_id", "analysis_label"]):
        for fold, heldout in rows.groupby("cv_fold"):
            training = rows.loc[rows["cv_fold"].ne(fold)]
            prediction = float(training["human_rate"].mean())
            scores = item_log_score(heldout["human_rate"], pd.Series(prediction, index=heldout.index))
            for index, score in scores.items():
                records.append({
                    "analysis_dataset_id": dataset_id,
                    "dataset": label,
                    "analysis_unit_id": heldout.loc[index, "analysis_unit_id"],
                    "human_rate": float(heldout.loc[index, "human_rate"]),
                    "prediction": prediction,
                    "oof_log_score": float(score),
                })
    raw = pd.DataFrame.from_records(records)
    by_dataset = raw.groupby(["analysis_dataset_id", "dataset"], as_index=False).agg(
        N=("analysis_unit_id", "size"),
        mean_oof_log_score=("oof_log_score", "mean"),
        mean_prediction=("prediction", "mean"),
        mean_human_rate=("human_rate", "mean"),
    )
    return by_dataset, float(by_dataset["mean_oof_log_score"].mean())


def _plot_log_scores(
    summary: pd.DataFrame, output: Path, analysis_label: str
) -> None:
    ordered = summary.sort_values("balanced_mean_oof_log_score", ascending=True)
    colors = ["#777777" if label == "Dataset base rate" else "#4C78A8" for label in ordered["model"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ordered["model"], ordered["balanced_mean_oof_log_score"], color=colors)
    ax.set_xlabel("Balanced out-of-fold mean log score (higher is better)")
    ax.set_title(f"{analysis_label} sampled-prefix model comparison")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_correlations(
    correlations: pd.DataFrame,
    output: Path,
    linking_correlations: pd.DataFrame | None = None,
    analysis_label: str = "Preliminary",
) -> None:
    if linking_correlations is not None:
        labels = [column for column in linking_correlations.columns if column != "Dataset"]
        datasets = linking_correlations["Dataset"].astype(str).tolist()
        values = linking_correlations[labels].apply(pd.to_numeric, errors="coerce").to_numpy()
        fig, ax = plt.subplots(figsize=(13, 6.5))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#D9D9D9")
        image = ax.imshow(
            np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1, aspect="auto"
        )
        ax.set_xticks(
            range(len(LINKING_SHORT_LABELS)), LINKING_SHORT_LABELS,
            rotation=35, ha="right"
        )
        ax.set_yticks(range(len(datasets)), datasets)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                label = "N/A" if np.isnan(values[row, column]) else f"{values[row, column]:.2f}"
                ax.text(column, row, label, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Pearson r")
        ax.set_title(f"{analysis_label} model–human correlations by dataset")
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)
        return

    columns = [
        ("top_k", "set", "Top-K Set"),
        ("top_k", "ordering", "Ordering"),
        ("top_k", "conjunction", "Top-K Conj."),
        ("top_k", "disjunction", "Top-K Disj."),
        ("top_p", "set", "Top-p Set"),
        ("top_p", "conjunction", "Top-p Conj."),
        ("top_p", "disjunction", "Top-p Disj."),
    ]
    datasets = correlations.loc[correlations["variant"].eq("top_k"), [
        "analysis_dataset_id", "dataset"
    ]].sort_values("analysis_dataset_id")
    matrix = []
    for dataset_id in datasets["analysis_dataset_id"]:
        values = []
        for variant, structure, _ in columns:
            row = correlations.loc[
                correlations["variant"].eq(variant)
                & correlations["analysis_dataset_id"].eq(dataset_id)
            ].iloc[0]
            values.append(float(row[f"{structure}_pearson_r"]))
        matrix.append(values)
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), [label for _, _, label in columns], rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)), datasets["dataset"])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson r")
    ax.set_title(f"{analysis_label} model–human correlations by dataset")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_linking_log_scores(
    log_scores: pd.DataFrame, output: Path, analysis_label: str
) -> None:
    labels = [column for column in log_scores.columns if column != "Dataset"]
    datasets = log_scores["Dataset"].astype(str).tolist()
    values = log_scores[labels].apply(pd.to_numeric, errors="coerce").to_numpy()
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Linking-structure log-score table has no finite values")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    # A few catastrophic scores near -9 otherwise compress the much denser
    # -1 to -4 range into nearly identical warm colors.  Robust limits improve
    # discrimination while the annotations continue to show exact values.
    color_min, color_max = np.nanpercentile(finite, [10, 90])
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#D9D9D9")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap,
        vmin=float(color_min), vmax=float(color_max), aspect="auto",
    )
    ax.set_xticks(
        range(len(LINKING_SHORT_LABELS)), LINKING_SHORT_LABELS,
        rotation=35, ha="right",
    )
    ax.set_yticks(range(len(datasets)), datasets)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            label = "N/A" if np.isnan(values[row, column]) else f"{values[row, column]:.2f}"
            ax.text(column, row, label, ha="center", va="center", fontsize=8)
    fig.colorbar(
        image, ax=ax, extend="both",
        label="Mean proper log score (higher is better)",
    )
    ax.set_title(f"{analysis_label} model–human log scores by dataset")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot-dir", type=Path,
        help="Source manifest used to derive dynamic coverage notes.",
    )
    parser.add_argument(
        "--linking-correlations", type=Path,
        help="Optional wide correlation table including direct linking baselines.",
    )
    parser.add_argument(
        "--linking-log-scores", type=Path,
        help="Optional wide proper-log-score table including direct linking baselines.",
    )
    parser.add_argument(
        "--analysis-label", default="Preliminary",
        help="Label used in plot and report titles (for example, 'Full').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv_dir = args.results_dir / "cv_results"
    oof = pd.read_csv(cv_dir / "oof_predictions.csv")
    balanced = pd.read_csv(cv_dir / "oof_balanced_log_scores_by_structure.csv")
    correlations = pd.read_csv(cv_dir / "correlations.csv")
    selections = pd.read_csv(cv_dir / "fold_selections.csv")
    base_by_dataset, base_score = _dataset_base_rate(oof)

    model_rows = balanced.copy()
    model_rows["model"] = model_rows["variant"].str.replace("_", "-", regex=False) + " " + model_rows["structure"]
    # Ordering is boundary-independent, so show it once.
    model_rows = model_rows.loc[~(
        model_rows["variant"].eq("top_p") & model_rows["structure"].eq("ordering")
    )].copy()
    base_row = pd.DataFrame([{
        "variant": "baseline", "structure": "dataset_base_rate",
        "balanced_mean_oof_log_score": base_score,
        "dataset_count": base_by_dataset["analysis_dataset_id"].nunique(),
        "total_unit_count": base_by_dataset["N"].sum(),
        "model": "Dataset base rate",
    }])
    advisor_scores = pd.concat([base_row, model_rows], ignore_index=True)
    advisor_scores["delta_vs_dataset_base_rate"] = (
        advisor_scores["balanced_mean_oof_log_score"] - base_score
    )
    selection_counts = selections.groupby(
        ["variant", "selected_boundary"], as_index=False
    ).size().rename(columns={"size": "fold_count"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    advisor_scores.to_csv(args.output_dir / "advisor_balanced_oof_log_scores.csv", index=False)
    base_by_dataset.to_csv(args.output_dir / "advisor_dataset_base_rate_scores.csv", index=False)
    correlations.to_csv(args.output_dir / "advisor_correlations.csv", index=False)
    selection_counts.to_csv(args.output_dir / "advisor_boundary_selection_counts.csv", index=False)
    _plot_log_scores(
        advisor_scores,
        args.output_dir / "advisor_balanced_oof_log_scores.png",
        args.analysis_label,
    )
    linking_correlations = (
        pd.read_csv(args.linking_correlations)
        if args.linking_correlations is not None
        else None
    )
    _plot_correlations(
        correlations,
        args.output_dir / "advisor_correlations_heatmap.png",
        linking_correlations=linking_correlations,
        analysis_label=args.analysis_label,
    )
    if args.linking_log_scores is not None:
        _plot_linking_log_scores(
            pd.read_csv(args.linking_log_scores),
            args.output_dir / "advisor_log_scores_heatmap.png",
            args.analysis_label,
        )

    best = model_rows.sort_values("balanced_mean_oof_log_score", ascending=False).iloc[0]
    validation = json.loads((args.results_dir / "score_validation.json").read_text())
    prompt_count = int(validation["expected_prompts"])
    unit_count = int(
        oof.loc[oof["variant"].eq("top_k"), ["analysis_dataset_id", "analysis_unit_id"]]
        .drop_duplicates().shape[0]
    )
    source_row_count = "unknown"
    coverage_notes = []
    if args.snapshot_dir is not None:
        source = pd.read_csv(args.snapshot_dir / "source_rows.csv")
        source_row_count = f"{len(source):,}"
        novel_contexts = source.loc[
            source["dataset_family"].eq("novel_focus"), "context_id"
        ].nunique()
        g18_prompts = source.loc[
            source["dataset_family"].eq("hu_2023_benchmark")
            & source["dataset"].eq("g18"), "prompt_id"
        ].nunique()
        coverage_notes = [
            f"- {novel_contexts} novel-focus contexts are represented.",
            f"- Hu g18 contains {g18_prompts} distinct scored prompts in this analysis.",
        ]
    k_counts = selection_counts.loc[selection_counts["variant"].eq("top_k")]
    p_counts = selection_counts.loc[selection_counts["variant"].eq("top_p")]
    k_text = ", ".join(
        f"K={row.selected_boundary:g} in {int(row.fold_count)} fold(s)"
        for row in k_counts.itertuples()
    )
    p_text = ", ".join(
        f"p={row.selected_boundary:g} in {int(row.fold_count)} fold(s)"
        for row in p_counts.itertuples()
    )
    novel = correlations.loc[correlations["analysis_dataset_id"].eq("novel_focus")]
    novel_values = novel[[
        f"{structure}_pearson_r" for structure in STRUCTURES
    ]].to_numpy(dtype=float)
    novel_min, novel_max = np.nanmin(novel_values), np.nanmax(novel_values)
    rnx = oof.loc[oof["variant"].eq("top_k")].copy()
    rnx["matched_item"] = rnx["analysis_unit_id"].str.split("::").str[-1]
    matched_deltas = []
    for baseline_id, only_id, label in (
        ("rnx_esi", "rnx_eonly", "Eonly"),
        ("rnx_estrong", "rnx_eonlystrong", "Eonlystrong"),
    ):
        baseline = rnx.loc[
            rnx["analysis_dataset_id"].eq(baseline_id),
            ["matched_item", "human_rate"],
        ]
        only = rnx.loc[
            rnx["analysis_dataset_id"].eq(only_id),
            ["matched_item", "human_rate"],
        ]
        matched = baseline.merge(
            only, on="matched_item", suffixes=("_baseline", "_only"),
            validate="one_to_one",
        )
        matched_deltas.append(
            f"{label} by {float((matched['human_rate_only'] - matched['human_rate_baseline']).mean()):.2f}"
        )
    is_preliminary = args.analysis_label.strip().lower() == "preliminary"
    status_text = (
        f"These results are an advisor preview based on {prompt_count} of 360 Qwen prompts\n"
        f"({source_row_count} canonical rows; {unit_count} analysis units). They are not\n"
        "the final paper analysis."
        if is_preliminary
        else f"These results use all {prompt_count} Qwen prompts "
        f"({source_row_count} canonical rows; {unit_count} analysis units)."
    )
    completion_note = (
        "- Prompt completion order is hash-based and approximately randomized, but the\n"
        "  subset is not the complete planned sample.\n"
        "- Final conclusions must be replaced with results from all 360 prompts."
        if is_preliminary
        else "- All planned Qwen prompts are included."
    )
    coverage_heading = "Coverage caveats" if is_preliminary else "Coverage"
    markdown = f"""# {args.analysis_label} sampled-prefix results

{status_text}

## Main results

- Best proposed structure: **{best['structure']}** with balanced OOF mean log score **{best['balanced_mean_oof_log_score']:.3f}**.
- Fold-safe dataset base-rate score: **{base_score:.3f}**.
- Difference between the best proposed structure and base rate: **{best['balanced_mean_oof_log_score'] - base_score:.3f}** (negative is worse).
- Top-K selected **{k_text}**; selections at K=100 are at the maximum tested value.
- Top-p selected **{p_text}**.
- Novel-focus correlations range from approximately **{novel_min:.2f} to {novel_max:.2f}** across structures, while most scalar-dataset correlations are weak or inconsistent.
- Matched R&X prompts produce identical model predictions even though mean human exclusion increases in {" and ".join(matched_deltas)}.

## Interpretation

Ordering is the strongest of the proposed structures, but none beats a simple
fold-safe condition-specific base-rate predictor in proper log score. Extreme
0/1 Monte Carlo probabilities produce large penalties when the model disagrees
with non-extreme human rates. This suggests that calibration/noise and the
Top-K search range require discussion before the final analysis is frozen.

## {coverage_heading}

{chr(10).join(coverage_notes)}
{completion_note}
"""
    dataset_spearman_path = args.results_dir / "linking_structure_tables/spearman_by_dataset_and_linking_structure.csv"
    if dataset_spearman_path.exists():
        from build_linking_structure_tables import _markdown_table
        markdown += "\n\n## Spearman: Hu datasets and R&X conditions\n\n"
        markdown += "Hu ranks scales within each dataset after template aggregation. R&X ranks 60 items within each condition. The units match Pearson; ties receive average ranks.\n\n"
        markdown += _markdown_table(pd.read_csv(dataset_spearman_path))
    spearman_report = args.results_dir / "spearman/SPEARMAN.md"
    if spearman_report.exists():
        markdown += "\n\n" + spearman_report.read_text(encoding="utf-8").replace("# Within-context Spearman", "## Within-context Spearman", 1)
    (args.output_dir / "ADVISOR_SUMMARY.md").write_text(markdown, encoding="utf-8")
    print(f"[complete] wrote advisor summary to {args.output_dir}")


if __name__ == "__main__":
    main()
