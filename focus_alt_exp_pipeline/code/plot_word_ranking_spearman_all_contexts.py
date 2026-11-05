"""Export the viewer's 16 word-ranking plots as a 4-by-4 appendix figure.

Uses the embedded sampling-array scores in the current offline viewer. Checks
all ranks against their scores and the human source, recomputes every displayed
coefficient, and exports PDF/PNG/SVG, plotted data, a caption, and provenance.
No rescoring, sampling, fitting, or statistical inference is performed.

Run from any directory with Python, numpy, pandas, and matplotlib installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/focus-word-ranking-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE.parent
DEFAULT_VIEWER = PIPELINE / "results_viewer/index.html"
HUMAN_FILE = PIPELINE / "human_exp_data/sca_dataframe.csv"
STEM = "word_ranking_spearman_all_contexts"
TITLE = "Word-ranking agreement across contexts"
FIGURE_SIZE = (7.8, 8.85)
SUBPLOT_LAYOUT = dict(left=.06, right=.989, bottom=.058, top=.913,
                      wspace=.37, hspace=.48)
POINT_AREA = 19
LABEL_OFFSET = 3
BLUE, INK, GRID, REFERENCE = "#4275D6", "#26323C", "#E2E7EE", "#899AAF"
CAPTION = (
    "Word-ranking agreement in all 16 focus contexts. Each panel plots the six "
    "tested alternatives, with human relevance rank (trigger_relevance) on the "
    "horizontal axis and Qwen2-7B rank on the vertical axis. Model ranks are "
    "calculated within each context from the sampling-array log probabilities "
    "used in the HTML viewer. Rank 0 is highest; tied model scores receive "
    "average ranks (fan and heater in hot both have rank 0.5). The dashed "
    "diagonal indicates identical human and model ranks. Panel annotations "
    "report Spearman's rho across the six words, rounded to three decimal "
    "places. Contexts are arranged alphabetically."
)


def collect(viewer: Path):
    html = viewer.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="results-data" type="application/json">(.*?)</script>',
        html, re.S,
    )
    if match is None:
        raise ValueError("Viewer has no embedded results-data")
    data = json.loads(match.group(1))
    ranks = pd.DataFrame(data["spearman"]["word_paired_ranks"])
    saved = pd.DataFrame(data["spearman"]["word_spearman_by_context"])
    contexts = sorted(data["contexts"])
    assert len(contexts) == len(set(contexts)) == 16
    assert set(ranks.context) == set(saved.context) == set(contexts)
    assert len(ranks) == 96 and not ranks.duplicated(["context", "word"]).any()
    assert len(saved) == 16 and not saved.context.duplicated().any()
    assert ranks.score_source.eq("sampling_array").all()
    assert saved.score_source.eq("sampling_array").all()
    assert saved.status.eq("defined").all() and saved.n.eq(6).all()
    assert ranks.model.eq("Qwen2-7B").all()

    human = pd.read_csv(HUMAN_FILE).rename(
        columns={"story": "context", "cleaned_trigger": "word"}
    )
    human = human[["context", "word", "trigger_relevance"]].drop_duplicates()
    for col in ["context", "word"]:
        human[col] = human[col].str.strip().str.lower()
    human = human.rename(columns={"trigger_relevance": "source_human_rank"})
    joined = ranks.merge(human, on=["context", "word"], how="outer",
                         validate="one_to_one", indicator=True)
    assert joined._merge.eq("both").all()
    np.testing.assert_array_equal(joined.human_rank, joined.source_human_rank)

    summaries = []
    for context in contexts:
        rows = ranks.loc[ranks.context.eq(context)]
        assert len(rows) == 6 and set(rows.human_rank) == set(range(6))
        assert rows.prompt_id.nunique() == 1
        assert np.isfinite(rows.model_value.to_numpy()).all()
        calculated = rows.model_value.rank(method="average", ascending=False) - 1
        np.testing.assert_array_equal(rows.model_rank, calculated)
        rho = rows.human_rank.corr(rows.model_rank)
        expected = saved.loc[saved.context.eq(context), "spearman_rho"].item()
        np.testing.assert_allclose(rho, expected, rtol=0, atol=1e-12)
        summaries.append({"context": context, "n_words": 6,
                          "spearman_rho": rho, "display_rho": f"{rho:.3f}"})
    ranks = ranks.sort_values(["context", "human_rank"]).reset_index(drop=True)
    return data, ranks, pd.DataFrame(summaries)


def plot(ranks: pd.DataFrame, summaries: pd.DataFrame, output_dir: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": True, "mathtext.default": "regular",
    })
    fig, axes = plt.subplots(4, 4, figsize=FIGURE_SIZE)
    fig.subplots_adjust(**SUBPLOT_LAYOUT)
    fig.suptitle(TITLE, x=.53, y=.983, fontsize=11, fontweight="bold")
    labels = []
    for ax, summary in zip(axes.flat, summaries.to_dict("records")):
        context = summary["context"]
        rows = ranks.loc[ranks.context.eq(context)]
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-.24, 5.24)
        ax.set_ylim(-.24, 5.46)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.tick_params(labelsize=6, length=0, pad=3)
        ax.grid(color=GRID, linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.plot([0, 5], [0, 5], color=REFERENCE, linewidth=.75,
                linestyle=(0, (3, 3)), zorder=1)
        ax.scatter(rows.human_rank, rows.model_rank, s=POINT_AREA, color=BLUE,
                   edgecolors="white", linewidths=.35, zorder=3)
        ax.set_title(context, loc="left", fontsize=8, fontweight="bold", pad=5)
        ax.set_title(rf"$\rho = {summary['display_rho']}$", loc="right",
                     fontsize=7.5, color=BLUE, pad=5)
        for row in rows.itertuples():
            right_side = row.human_rank > 3
            text = ax.annotate(
                row.word, xy=(row.human_rank, row.model_rank),
                xytext=(-LABEL_OFFSET if right_side else LABEL_OFFSET, LABEL_OFFSET),
                textcoords="offset points", ha="right" if right_side else "left",
                va="bottom", fontsize=6.1, color=INK, clip_on=False, zorder=4,
                path_effects=[path_effects.withStroke(linewidth=1.7,
                                                     foreground="white")],
            )
            labels.append((context, text))
    fig.supxlabel("Human rank (0 = highest)", x=.53, y=.018, fontsize=9)
    fig.supylabel("Model rank (0 = highest)", x=.013, y=.52, fontsize=9)

    # Check the actual rendered text boxes, rather than assuming labels fit.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for index, (context, label) in enumerate(labels):
        bbox = label.get_window_extent(renderer)
        assert fig.bbox.contains(bbox.x0, bbox.y0)
        assert fig.bbox.contains(bbox.x1, bbox.y1)
        for other_context, other in labels[:index]:
            if bbox.overlaps(other.get_window_extent(renderer)):
                raise ValueError(f"Overlapping labels: {context}/{label.get_text()} "
                                 f"and {other_context}/{other.get_text()}")
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(output_dir / f"{STEM}.{extension}", dpi=400, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, default=DEFAULT_VIEWER)
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY / "output/pdf")
    args = parser.parse_args()
    data, ranks, summaries = collect(args.viewer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(ranks, summaries, args.output_dir)
    ranks.merge(summaries[["context", "spearman_rho"]], on="context",
                validate="many_to_one").to_csv(
                    args.output_dir / f"{STEM}_ranks.csv", index=False)
    summaries.to_csv(args.output_dir / f"{STEM}_correlations.csv", index=False)
    (args.output_dir / f"{STEM}_caption.txt").write_text(CAPTION + "\n")
    latex_caption = CAPTION.replace("Spearman's rho", r"Spearman's $\rho$")
    (args.output_dir / f"{STEM}_inclusion.tex").write_text(
        "% Requires \\usepackage{graphicx}. Copy the PDF to the manuscript's\n"
        "% figures directory; figure* spans both columns in a two-column paper.\n"
        "\\begin{figure*}[p]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{figures/{STEM}.pdf}}\n"
        f"  \\caption{{{latex_caption}}}\n"
        "  \\label{fig:appendix-word-ranking-spearman}\n"
        "\\end{figure*}\n"
    )
    provenance = {
        "run": data["run"]["id"],
        "viewer": str(args.viewer.resolve()),
        "viewer_sha256": hashlib.sha256(args.viewer.read_bytes()).hexdigest(),
        "human_source": str(HUMAN_FILE),
        "human_source_sha256": hashlib.sha256(HUMAN_FILE.read_bytes()).hexdigest(),
        "source_fields": ["spearman.word_paired_ranks",
                          "spearman.word_spearman_by_context"],
        "score_source": "sampling_array",
        "n_contexts": 16, "words_per_context": 6, "n_points": 96,
        "panel_order": summaries.context.tolist(),
        "rank_convention": "Descending, zero-based, average ranks for ties",
        "coefficient": "Pearson correlation of human and model rank columns",
        "display_precision": 3,
        "validation": [
            "All 96 human ranks match sca_dataframe.csv trigger_relevance",
            "All 96 model ranks recomputed from viewer sampling-array log scores",
            "All 16 recomputed coefficients match viewer values within 1e-12",
            "All word label boxes fit the figure and do not overlap each other",
        ],
        "title": TITLE,
        "figure_inches": list(FIGURE_SIZE), "png_dpi": 400,
        "subplot_layout": SUBPLOT_LAYOUT,
        "point_area_pt_squared": POINT_AREA,
        "label_offset_pt": LABEL_OFFSET,
        "script": str(Path(__file__).resolve()),
    }
    (args.output_dir / f"{STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print("Verified 96 ranks and 16 Spearman coefficients against the viewer and human source.")
    print(summaries.to_string(index=False))
    print(f"Exported PDF, PNG, SVG, data, caption, LaTeX inclusion, and provenance to {args.output_dir}")


if __name__ == "__main__":
    main()
