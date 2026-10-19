"""Recreate Option 2 with all 16 contexts, using the original held-out results.

Run with Python plus numpy, pandas, and matplotlib. The default output directory
is <repository>/output/pdf; use --output-dir to override it. Original seven-context
figures and their data are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path

from plot_options import BLUE, GRAY, LIGHT, OUT, PIPELINE, SOURCE, style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

NAME = "option_2_context_improvements_all_16"
CAPTION = (
    "Context-level changes in exclusion correlation after applying Set Top-K, "
    "for all 16 contexts (480 ordered trigger-query pairs). Points show Set Top-K "
    "minus No linking for (A) Spearman and (B) Pearson exclusion correlations, "
    "each calculated across 30 equally weighted ordered pairs. Blue diamonds "
    "identify perfect model-human word-ranking agreement (bakery and fridge); "
    "blue circles identify the five contexts with word-ranking Spearman = .943 "
    "(rounded); gray circles identify the nine contexts with word-ranking "
    "Spearman <= .9. Contexts are ordered alphabetically. Positive values favor "
    "Set Top-K. Both panels use identical horizontal limits. Values are descriptive "
    "point differences, not significance tests or confidence intervals; predictions "
    "are unchanged from the corrected held-out evaluation."
)


def collect():
    word_path = SOURCE / "word_spearman_by_context.csv"
    pred_path = SOURCE / "negation_paired_ranks.csv"
    words = pd.read_csv(word_path).set_index("context")
    predictions = pd.read_csv(pred_path)
    records = []
    for context, group in predictions.groupby("context", sort=True):
        baseline = group[group.structure.eq("No linking structure")].sort_values("item_id")
        linking = group[group.structure.eq("set") & group.variant.eq("top_k")].sort_values("item_id")
        assert baseline.item_id.nunique() == linking.item_id.nunique() == 30
        assert len(baseline) == len(linking) == 30
        assert baseline.item_id.tolist() == linking.item_id.tolist()
        assert np.array_equal(baseline.human_rate.to_numpy(), linking.human_rate.to_numpy())
        row = {"context": context, "word_rho": words.loc[context, "spearman_rho"]}
        for prefix, frame in [("no_linking", baseline), ("set_top_k", linking)]:
            row[prefix + "_pearson"] = frame.model_value.corr(frame.human_rate)
            row[prefix + "_spearman"] = frame.model_value.rank().corr(frame.human_rate.rank())
        for metric in ["pearson", "spearman"]:
            row["delta_" + metric] = row["set_top_k_" + metric] - row["no_linking_" + metric]
        records.append(row)
    data = pd.DataFrame(records).set_index("context").sort_index()
    assert len(data) == 16 and set(data.index) == set(words.index)
    assert np.isfinite(data.to_numpy()).all()
    assert data.word_rho.gt(.9).sum() == 7
    assert np.isclose(data.word_rho, 1, rtol=0, atol=1e-12).sum() == 2
    # Verify every coefficient against the saved data underlying the original PDF.
    saved_path = OUT / "figure_data.csv"
    saved = pd.read_csv(saved_path).set_index("context").sort_index()
    pd.testing.assert_frame_equal(data, saved[data.columns], rtol=0, atol=1e-12)
    return data, [word_path, pred_path, saved_path]


def plot(data, output_dir, model_label="Set Top-K", output_name=NAME,
         xlimits=(-.30, .38), xticks=(-.2, -.1, 0, .1, .2, .3)):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 7.7), sharey=True)
    fig.subplots_adjust(left=.17, right=.975, top=.845, bottom=.19, wspace=.19)
    fig.suptitle(f"Where does {model_label} improve exclusion prediction?",
                 y=.975, fontsize=12, fontweight="bold")
    fig.text(.5, .936, "All 16 contexts / 480 ordered trigger-query pairs",
             ha="center", fontsize=9, color=GRAY)
    for ax, metric, title in zip(axes, ["spearman", "pearson"],
                                 ["A  Change in Spearman", "B  Change in Pearson"]):
        style(ax)
        ax.set_title(title, loc="left", pad=12)
        ax.axvline(0, color=GRAY, lw=1, ls=(0, (3, 3)))
        for y, (_, row) in enumerate(data.iterrows()):
            value = row["delta_" + metric]
            perfect = np.isclose(row.word_rho, 1, rtol=0, atol=1e-12)
            color = BLUE if row.word_rho > .9 else GRAY
            ax.plot([0, value], [y, y], color=LIGHT, lw=1.5, zorder=2)
            ax.scatter(value, y, s=41 if perfect else 33,
                       marker="D" if perfect else "o", color=color,
                       edgecolors="white", lw=.5, zorder=3)
            ax.annotate(f"{value:+.3f}", (value, y),
                        xytext=(8 if value >= 0 else -8, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center",
                        fontsize=8, color=color)
        ax.set_xlim(*xlimits)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: "0" if abs(x) < 1e-10 else f"{x:+.2f}"))
        ax.set_ylim(len(data) - .35, -.65)
        ax.set_yticks(range(len(data)), [c.title() for c in data.index])
        ax.set_xlabel(f"{model_label} minus No linking", labelpad=9)
    handles = [
        Line2D([], [], marker="o", ls="", color=GRAY, markersize=5,
               label="Word-ranking $\\rho\\leq .9$"),
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=5,
               label="Word-ranking $\\rho=.943$"),
        Line2D([], [], marker="D", ls="", color=BLUE, markersize=5,
               label="Word-ranking $\\rho=1$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .075),
               ncol=3, frameon=False, fontsize=8.5, columnspacing=1.4)
    fig.text(.5, .032,
             f"Positive values favor {model_label}. Each correlation uses 30 pairs; both panels share the same scale.",
             ha="center", fontsize=7.5, color=GRAY)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(output_dir / f"{output_name}.{ext}", dpi=220 if ext == "png" else 300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parent / "output/pdf")
    args = parser.parse_args()
    data, sources = collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(data, args.output_dir)
    data.to_csv(args.output_dir / f"{NAME}.csv")
    (args.output_dir / f"{NAME}_caption.txt").write_text(CAPTION + "\n")
    (args.output_dir / f"{NAME}_provenance.json").write_text(json.dumps({
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "script": str(Path(__file__).resolve()),
        "selection": "All 16 contexts, alphabetical order; 30 pairs per context, 480 total.",
        "evaluation": "Set Top-K minus No linking within-context correlations. Spearman uses average ranks for ties. Original corrected held-out predictions; no fitting or resampling.",
        "validation": "All 16 contexts and every coefficient match the original saved figure data within absolute tolerance 1e-12.",
        "horizontal_limits": [-.30, .38],
    }, indent=2) + "\n")
    print("Verified all 16 contexts against original figure data; exported PDF, PNG, SVG, data, caption, and provenance.")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
