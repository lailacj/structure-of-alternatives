"""Plot exclusion Spearman by model with all context labels and model means.

Recompute the 80 context coefficients from the corrected saved predictions,
check the saved context and mean tables, and export PDF/PNG/SVG plus plot data,
caption, LaTeX inclusion, and provenance. Existing figures are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/focus-word-ranking-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parents[1]
RESULTS = PIPELINE / "results/set_variant_qwen_corrected_rnx_contrastive_2026-10-23"
STEM = "focus_exclusion_spearman_labeled_contexts"
SELECTORS = [
    ("No linking structure", "direct", "No linking"),
    ("set", "top_k", "Set"),
    ("ordering", "top_k", "Ordering"),
    ("conjunction", "top_k", "Conjunction"),
    ("disjunction", "top_k", "Disjunction"),
]
MEAN, POINT, TEXT, GRID = "#174E72", "#ADB8C1", "#59656E", "#E9EDF0"
TITLE = "Focus study: exclusion rankings"
FIGURE_SIZE = (3.5, 4.8)
ROW_STEP = 1.35
LANES = (-.24, .24, -.40, .40, -.56, .56)
CAPTION = (
    "Within-context agreement between model predictions and human exclusion "
    "rankings in the focus alternative study. Each labeled gray point shows "
    "Spearman's rho across the 30 ordered trigger-query pairs in one context. "
    "Larger blue points and numerical annotations show the equally weighted "
    "mean across all 16 contexts for each model. Vertical offsets separate "
    "labels and have no quantitative meaning. Linking predictions use the "
    "saved Top-K evaluation with boundaries selected on training folds and "
    "predictions evaluated on held-out items. Model names are shortened for "
    "display. All five models have defined correlations in all 16 contexts. "
    "These are descriptive point estimates; the spread of context points "
    "does not represent a confidence interval."
)


def collect(results: Path):
    sources = [results / "spearman" / name for name in [
        "negation_paired_ranks.csv", "negation_spearman_by_context.csv",
        "mean_within_context_spearman.csv",
    ]]
    pairs, saved_contexts, saved_means = [pd.read_csv(p) for p in sources]
    rows, summaries = [], []
    human_reference = None
    expected_contexts = None
    for structure, variant, label in SELECTORS:
        selected = pairs.loc[pairs.structure.eq(structure) & pairs.variant.eq(variant)]
        assert len(selected) == 480 and selected.item_id.nunique() == 480
        assert selected.context.nunique() == 16
        human = selected.set_index("item_id").human_rate.sort_index()
        if human_reference is None:
            human_reference = human
            expected_contexts = set(selected.context)
        else:
            pd.testing.assert_series_equal(human_reference, human)
            assert set(selected.context) == expected_contexts
        coefficients = []
        for context, group in selected.groupby("context", sort=True):
            assert len(group) == group.item_id.nunique() == 30
            assert np.isfinite(group[["human_rate", "model_value"]]).all().all()
            rho = group.model_value.rank(method="average").corr(
                group.human_rate.rank(method="average"))
            assert np.isfinite(rho)
            saved = saved_contexts.loc[
                saved_contexts.structure.eq(structure) &
                saved_contexts.variant.eq(variant) & saved_contexts.context.eq(context)]
            assert len(saved) == 1
            np.testing.assert_allclose(rho, saved.spearman_rho.item(), rtol=0, atol=1e-12)
            rows.append(dict(model=label, structure=structure, variant=variant,
                             context=context, n_pairs=30, spearman_rho=rho))
            coefficients.append(rho)
        average = float(np.mean(coefficients))
        saved = saved_means.loc[saved_means.measure.eq("negation") &
                                saved_means.structure.eq(structure) &
                                saved_means.variant.eq(variant)]
        assert len(saved) == 1 and saved.valid_contexts.item() == 16
        np.testing.assert_allclose(average, saved.mean_within_context_spearman.item(),
                                   rtol=0, atol=1e-12)
        summaries.append(dict(model=label, n_contexts=16, n_pairs=480,
                              mean_within_context_spearman=average))
    return pd.DataFrame(rows), pd.DataFrame(summaries), sources


def draw(contexts: pd.DataFrame, summary: pd.DataFrame, output: Path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=.26, right=.975, top=.94, bottom=.15)
    fig.suptitle(TITLE, x=.58, y=.99, fontsize=10.5, fontweight="bold")
    centers = np.arange(5) * ROW_STEP
    ax.set_xlim(-.08, 1.02)
    ax.set_ylim(centers[-1] + .64, -.64)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
    ax.set_yticks(centers, [x[2] for x in SELECTORS], fontsize=9.5)
    ax.set_xlabel(r"Within-context Spearman $\rho$", fontsize=10.5, labelpad=4)
    ax.tick_params(length=0, pad=5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=.65, zorder=0)
    ax.axvline(0, color="#BEC7CE", linewidth=.8, zorder=0)
    for center in (centers[:-1] + centers[1:]) / 2:
        ax.axhline(center, color="#EDF0F2", linewidth=.65, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    mean_labels = []
    for center, row in zip(centers, summary.itertuples()):
        ax.scatter(row.mean_within_context_spearman, center, s=40, color=MEAN,
                   edgecolors="white", linewidths=.7, zorder=4)
        mean_labels.append(ax.annotate(
            f"{row.mean_within_context_spearman:.3f}",
            (row.mean_within_context_spearman, center), xytext=(5, 0),
            textcoords="offset points", va="center", fontsize=9.5,
            fontweight="bold", color=MEAN, zorder=5,
        ))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_labels, layout = [], []
    for center, (_, _, model) in zip(centers, SELECTORS):
        occupied = []
        group = contexts.loc[contexts.model.eq(model)].sort_values(
            ["spearman_rho", "context"])
        for row in group.itertuples():
            label = None
            candidates = [(offset, side) for offset in LANES
                          for side in (row.spearman_rho < .84, row.spearman_rho >= .84)]
            # Keep "bag" to the right so it cannot read as part of "cut".
            if model == "No linking" and row.context == "bag":
                candidates = [(offset, True) for offset in LANES]
            for offset, align_left in candidates:
                y = center + offset
                annotation = ax.annotate(
                    row.context, (row.spearman_rho, y),
                    xytext=(2.5 if align_left else -2.5, 0), textcoords="offset points",
                    ha="left" if align_left else "right", va="center",
                    fontsize=5.4, color=TEXT, zorder=3,
                )
                text_box = annotation.get_window_extent(renderer)
                point_x, point_y = ax.transData.transform((row.spearman_rho, y))
                point_box = Bbox.from_extents(point_x - 2, point_y - 2,
                                               point_x + 2, point_y + 2)
                combined = Bbox.union([text_box, point_box])
                combined = Bbox.from_extents(combined.x0 - 1.1, combined.y0 - .5,
                                               combined.x1 + 1.1, combined.y1 + .5)
                if (fig.bbox.contains(combined.x0, combined.y0) and
                    fig.bbox.contains(combined.x1, combined.y1) and
                    all(not combined.overlaps(box) for box in occupied) and
                    all(not combined.overlaps(t.get_window_extent(renderer).padded(1))
                        for t in mean_labels)):
                    label = annotation
                    occupied.append(combined)
                    ax.scatter(row.spearman_rho, y, s=7, color=POINT,
                               edgecolors="white", linewidths=.25, zorder=2)
                    layout.append(dict(model=model, context=row.context,
                                       spearman_rho=row.spearman_rho, row_center=center,
                                       vertical_offset=offset,
                                       label_side="right" if align_left else "left"))
                    all_labels.append(annotation)
                    break
                annotation.remove()
            if label is None:
                raise ValueError(f"No collision-free label position for {model}/{row.context}")

    handles = [Line2D([], [], marker="o", linestyle="", color=POINT,
                      markersize=2.8, label="Context"),
               Line2D([], [], marker="o", linestyle="", color=MEAN,
                      markersize=5, label="Mean across contexts")]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.59, .008),
                        ncol=2, frameon=False, fontsize=6.5, handletextpad=.4, columnspacing=1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not legend.get_window_extent(renderer).overlaps(
        ax.xaxis.label.get_window_extent(renderer))
    assert len(all_labels) == 80
    for index, label in enumerate(all_labels + mean_labels):
        bbox = label.get_window_extent(renderer)
        assert fig.bbox.contains(bbox.x0, bbox.y0) and fig.bbox.contains(bbox.x1, bbox.y1)
        for other in (all_labels + mean_labels)[:index]:
            assert not bbox.overlaps(other.get_window_extent(renderer)), (
                label.get_text(), other.get_text())
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(output / f"{STEM}.{extension}", dpi=400, facecolor="white")
    plt.close(fig)
    return pd.DataFrame(layout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parent / "output/pdf")
    args = parser.parse_args()
    contexts, summary, sources = collect(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout = draw(contexts, summary, args.output_dir)
    contexts.merge(layout, on=["model", "context", "spearman_rho"], validate="one_to_one").to_csv(
        args.output_dir / f"{STEM}_contexts.csv", index=False)
    summary.to_csv(args.output_dir / f"{STEM}_means.csv", index=False)
    (args.output_dir / f"{STEM}_caption.txt").write_text(CAPTION + "\n")
    latex_caption = CAPTION.replace("Spearman's rho", r"Spearman's $\rho$")
    (args.output_dir / f"{STEM}_inclusion.tex").write_text(
        "% Requires \\usepackage{graphicx}; copy the PDF to figures/.\n"
        "\\begin{figure}[t]\n  \\centering\n"
        f"  \\includegraphics[width=\\columnwidth]{{figures/{STEM}.pdf}}\n"
        f"  \\caption{{{latex_caption}}}\n"
        "  \\label{fig:focus-exclusion-spearman-contexts}\n\\end{figure}\n")
    provenance = {
        "results_directory": str(args.results_dir.resolve()),
        "metric": "within-context exclusion Spearman",
        "model_selectors": SELECTORS, "title": TITLE,
        "context_weighting": "equal; each model mean averages 16 context coefficients",
        "n_context_coefficients": 80, "n_means": 5, "pairs_per_context": 30,
        "vertical_offsets": "label layout only; no quantitative meaning",
        "figure_inches": FIGURE_SIZE, "png_dpi": 400,
        "target_layout": "single column at 3.5 inches wide",
        "font_sizes_pt": {"context": 5.4, "mean": 9.5, "model": 9.5,
                          "x_axis_title": 10.5, "x_ticks": 8.5},
        "mean_point_area_pt_squared": 40,
        "validation": ["80 coefficients recomputed from saved predictions",
                       "80 coefficients match saved context table within 1e-12",
                       "5 means match saved mean table within 1e-12",
                       "All 80 context labels and 5 mean labels fit without overlap"],
        "sources": [{"path": str(p.resolve()),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "script": str(Path(__file__).resolve()),
    }
    (args.output_dir / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("Verified 80 context coefficients, 5 means, and all label positions.")
    print(f"Saved PDF, PNG, SVG, data, caption, LaTeX inclusion, and provenance to {args.output_dir}")


if __name__ == "__main__":
    main()
