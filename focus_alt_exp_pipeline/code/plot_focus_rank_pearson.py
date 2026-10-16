"""Shared Section 5.1/5.2 figure from the saved corrected focus predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
MODELS = ["No linking structure", "Set Top-K", "Ordering", "Conjunction Top-K", "Disjunction Top-K"]
SELECTORS = [("No linking structure", "direct"), ("set", "top_k"),
             ("ordering", "top_k"), ("conjunction", "top_k"), ("disjunction", "top_k")]
LABELS = ["No linking", "Set Top-K", "Ordering", "Conjunction Top-K", "Disjunction Top-K"]
MEAN_COLOR = "#174E72"
POOLED_COLOR = "#B55321"

CAPTION = """Ranking and numerical agreement with human focus-alternative exclusion judgments.
(A) Spearman correlations and (B) Pearson correlations for five predictors.
Small gray points show correlations across the 30 ordered trigger-query pairs
within each of 16 contexts; filled circles show their equally weighted mean.
Open diamonds in B show Pearson correlations pooled over all 480 pairs.
All five predictors have defined correlations in all 16 contexts. Set,
Conjunction, and Disjunction use Top-K boundaries selected on training folds;
linking predictions are evaluated on held-out items. Values are descriptive
point estimates; context-level points show variation, not confidence intervals.
Pearson correlation assesses numerical association, not absolute probability
accuracy. Predictions come from the corrected Qwen2-7B single-space run.
"""


def collect(results: Path):
    paths = [results / "advisor_summary/item_level_scatterplots/novel_focus_by_context/novel_focus_item_level_predictions.csv",
             results / "spearman/mean_within_context_spearman.csv",
             results / "spearman/negation_spearman_by_context.csv",
             results / "linking_structure_tables/correlations_by_dataset_and_linking_structure.csv"]
    predictions, saved_means, saved_contexts, pooled_table = [pd.read_csv(p) for p in paths]
    pooled_table = pooled_table.set_index("Dataset")
    contexts, summary = [], []
    reference = None
    for model, (structure, variant) in zip(MODELS, SELECTORS):
        items = predictions.loc[predictions.model.eq(model)].copy()
        assert len(items) == 480 and items.analysis_unit_id.nunique() == 480
        assert items.groupby("context_id").size().eq(30).all()
        assert items.context_id.nunique() == 16
        assert np.isfinite(items[["human_rate", "prediction"]]).all().all()
        human = items.set_index("analysis_unit_id").human_rate.sort_index()
        if reference is None:
            reference = human
        else:
            pd.testing.assert_series_equal(reference, human)
        rhos, rs = [], []
        for context, group in items.groupby("context_id", sort=True):
            r = group.prediction.corr(group.human_rate)
            rho = group.prediction.rank(method="average").corr(group.human_rate.rank(method="average"))
            assert np.isfinite([r, rho]).all()
            saved = saved_contexts.loc[saved_contexts.context.eq(context) & saved_contexts.structure.eq(structure)
                                       & saved_contexts.variant.eq(variant)]
            assert len(saved) == 1 and np.isclose(rho, saved.iloc[0].spearman_rho, atol=1e-12)
            contexts.append(dict(model=model, context=context, n_pairs=30, spearman_rho=rho, pearson_r=r))
            rhos.append(rho)
            rs.append(r)
        mean_rho = np.mean(rhos)
        saved = saved_means.loc[saved_means.measure.eq("negation") & saved_means.structure.eq(structure)
                                & saved_means.variant.eq(variant)]
        assert len(saved) == 1 and int(saved.iloc[0].valid_contexts) == 16
        assert np.isclose(mean_rho, saved.iloc[0].mean_within_context_spearman, atol=1e-12)
        pooled_r = items.prediction.corr(items.human_rate)
        assert np.isclose(pooled_r, pooled_table.loc["Novel Focus Alternative Study", model], atol=1e-12)
        summary.append(dict(model=model, mean_within_context_spearman=mean_rho,
                            mean_within_context_pearson=np.mean(rs), pooled_pearson=pooled_r,
                            n_contexts=16, n_pairs=480))
    return pd.DataFrame(contexts), pd.DataFrame(summary).set_index("model"), paths


def draw(contexts: pd.DataFrame, summary: pd.DataFrame, output: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), sharey=True)
    fig.suptitle("Focus Alternative Study:\nModel Predictions and Human Exclusion Correlations",
                 fontsize=11, fontweight="bold", y=.985, linespacing=1.15)
    fig.subplots_adjust(left=.235, right=.985, top=.79, bottom=.24, wspace=.20)
    for panel, (ax, context_metric, mean_metric, title, xlabel) in enumerate(zip(
        axes, ["spearman_rho", "pearson_r"],
        ["mean_within_context_spearman", "mean_within_context_pearson"],
        ["A  Exclusion rankings", "B  Exclusion rates"],
        [r"Spearman $\rho$", r"Pearson $r$"])):
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=15)
        ax.axvline(0, color="#B9C0C5", lw=.7, zorder=0)
        for i, model in enumerate(MODELS):
            row = contexts.loc[contexts.model.eq(model)].sort_values("context")
            jitter = np.random.default_rng(41).uniform(-.11, .11, len(row))
            ax.scatter(row[context_metric], i + jitter, s=13, color="#B9C1C7",
                       alpha=.65, edgecolors="none", zorder=2)
            value = summary.loc[model, mean_metric]
            mean_y = i - .19 if panel == 1 else i
            ax.scatter(value, mean_y, s=42, color=MEAN_COLOR, edgecolors="white", linewidths=.6, zorder=4)
            ax.annotate(f"{value:.3f}".lstrip("0"), (value, mean_y), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MEAN_COLOR,
                        bbox=dict(facecolor="white", edgecolor="none", pad=.5), zorder=5)
            if panel == 1:
                value = summary.loc[model, "pooled_pearson"]
                ax.scatter(value, i + .21, s=36, marker="D", facecolors="white",
                           edgecolors=POOLED_COLOR, linewidths=1.1, zorder=4)
                ax.annotate(f"{value:.3f}".lstrip("0"), (value, i + .21), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8, color=POOLED_COLOR,
                            bbox=dict(facecolor="white", edgecolor="none", pad=.5), zorder=5)
        ax.set_xlim(-.08, 1.02)
        ax.set_ylim(4.55, -.55)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
        ax.set_xlabel(xlabel, fontsize=9, labelpad=8)
        ax.set_yticks(range(5), LABELS)
        ax.tick_params(axis="both", length=0, pad=7, labelsize=8.5)
        ax.grid(axis="x", color="#EDF0F2", lw=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color="#B9C1C7", markersize=4, label="Individual context"),
               Line2D([], [], marker="o", ls="", color=MEAN_COLOR, markersize=6, label="Mean within context"),
               Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=POOLED_COLOR,
                      markersize=5, label="Pooled Pearson (B)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .025), ncol=3,
               frameon=False, fontsize=8, handletextpad=.5, columnspacing=1.4)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(output / f"focus_spearman_pearson.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PIPELINE / "results/set_variant_qwen_corrected_2026-09-15")
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parents[1] / "outputs/focus_results_5_1_5_2")
    args = parser.parse_args()
    contexts, summary, sources = collect(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(args.output_dir / "context_correlations.csv", index=False)
    summary.to_csv(args.output_dir / "summary_correlations.csv")
    draw(contexts, summary, args.output_dir)
    (args.output_dir / "caption.txt").write_text(CAPTION)
    (args.output_dir / "figure_inclusion.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth]{focus_spearman_pearson.pdf}\n"
        "\\caption{" + " ".join(CAPTION.split()) + "}\n"
        "\\label{fig:focus-rank-numerical}\n\\end{figure*}\n\n"
        "% Section 5.1: Figure~\\ref{fig:focus-rank-numerical}A\n"
        "% Section 5.2: Figure~\\ref{fig:focus-rank-numerical}B\n")
    provenance = {"results_directory": str(args.results_dir.resolve()), "models": MODELS,
                  "resampling": "none; descriptive point estimates", "context_weighting": "equal",
                  "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for p in sources]}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string())
    print(f"Saved figure, caption, inclusion snippet, values and provenance to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
