"""Simple bag histograms from saved human rates and No linking predictions.

Run with numpy, pandas and matplotlib. Save a two-panel figure, each panel
separately, and the plotted values/bin counts under <repository>/output/pdf.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[2]
SOURCE = PIPELINE / (
    "results/set_variant_qwen_corrected_2026-09-15/advisor_summary/"
    "item_level_scatterplots/novel_focus_by_context/novel_focus_item_level_predictions.csv"
)
NAME = "bag_human_no_linking_histograms"
BINS = {"human_rate": np.linspace(0, 1, 11),
        "prediction": np.linspace(0, .02, 11)}


def draw(ax, values, title, xlabel, color, bins):
    counts, _ = np.histogram(values, bins=bins)
    assert counts.sum() == len(values)
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=1.2)
    for left, right, count in zip(bins[:-1], bins[1:], counts):
        if count:
            ax.text((left + right) / 2, count + .45, str(count), ha="center", fontsize=12)
    ax.set(xlim=(bins[0], bins[-1]), ylim=(0, 32), xticks=np.linspace(bins[0], bins[-1], 6),
           yticks=np.arange(0, 31, 5), xlabel=xlabel,
           ylabel="Number of trigger-query pairs")
    ax.set_title(title + f" ({len(values)} pairs)", fontsize=16, fontweight="bold", pad=14)
    ax.text(.96, .95, f"Mean = {values.mean():.6f}\nSD = {values.std(ddof=0):.6f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            linespacing=1.5, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6EAEE", linewidth=.8)
    return counts


def save(fig, output_dir, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{name}.{ext}", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parent / "output/pdf")
    args = parser.parse_args()
    data = pd.read_csv(SOURCE)
    bag = data.loc[data.context_id.eq("bag") & data.model.eq("No linking structure"),
                   ["item_id", "context_id", "trigger", "query", "human_rate", "prediction"]]
    assert len(bag) == bag.item_id.nunique() == 30
    values = bag[["human_rate", "prediction"]]
    assert np.isfinite(values.to_numpy()).all()
    assert values.ge(0).all().all() and values.le(1).all().all()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("human_rate", "Human", "Human exclusion rate", "#687D8D", "bag_human_rate_histogram"),
        ("prediction", "No linking", "Model exclusion probability", "#176B92", "bag_no_linking_histogram"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8))
    fig.subplots_adjust(left=.075, right=.975, bottom=.19, top=.76, wspace=.28)
    fig.suptitle("Exclusion distributions in the bag context", fontsize=18,
                 fontweight="bold", y=.96)
    fig.text(.5, .065, "Different x-axis scales: human bins = 0.1; No linking bins = 0.002.",
             ha="center", fontsize=11, color="#53616E")
    fig.text(.5, .025, "Each pair counts once; SD calculated across all 30 pairs (divisor = 30).",
             ha="center", fontsize=10, color="#53616E")
    counts = {}
    for ax, (column, title, xlabel, color, _) in zip(axes, panels):
        counts[column] = draw(ax, bag[column], title, xlabel, color, BINS[column])
    save(fig, args.output_dir, NAME)
    for column, title, xlabel, color, name in panels:
        fig, ax = plt.subplots(figsize=(6, 5.4))
        fig.subplots_adjust(left=.16, right=.97, bottom=.19, top=.79)
        fig.suptitle("Bag context", fontsize=18, fontweight="bold", y=.97)
        draw(ax, bag[column], title, xlabel, color, BINS[column])
        fig.text(.5, .055, f"Bin width = {BINS[column][1]:g}",
                 ha="center", fontsize=11, color="#53616E")
        fig.text(.5, .015, "SD across all 30 pairs (divisor = 30)",
                 ha="center", fontsize=10, color="#53616E")
        save(fig, args.output_dir, name)
    # Additional view of the 25 predictions previously in the first 0.002 bin.
    zoom = bag.loc[bag.prediction.lt(.002)].copy()
    excluded = bag.loc[~bag.index.isin(zoom.index)]
    zoom_bins = np.linspace(0, .0002, 21)
    assert len(zoom) == 25 and len(excluded) == 5
    assert zoom.prediction.le(zoom_bins[-1]).all()
    fig, ax = plt.subplots(figsize=(8, 5.7))
    fig.subplots_adjust(left=.13, right=.96, bottom=.25, top=.77)
    fig.suptitle("Bag context: No linking, zoomed", fontsize=18,
                 fontweight="bold", y=.96)
    zoom_counts = draw(ax, zoom.prediction, "Low-probability predictions",
                       "Model exclusion probability", "#176B92", zoom_bins)
    ax.set(ylim=(0, 13), yticks=np.arange(0, 13, 2))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.text(.5, .13, "25 of 30 pairs shown; five predictions at 0.019163 are outside this view.",
             ha="center", fontsize=11, color="#53616E")
    fig.text(.5, .075, "Bin width = 0.00001; mean and SD describe these 25 pairs only.",
             ha="center", fontsize=10, color="#53616E")
    fig.text(.5, .025, "SD divisor = 25. Full-distribution plots are saved separately.",
             ha="center", fontsize=10, color="#53616E")
    save(fig, args.output_dir, "bag_no_linking_low_probability_zoom")
    zoom.to_csv(args.output_dir / "bag_no_linking_low_probability_zoom_values.csv", index=False)
    pd.DataFrame({"bin_left": zoom_bins[:-1], "bin_right": zoom_bins[1:],
                  "count": zoom_counts}).to_csv(
                      args.output_dir / "bag_no_linking_low_probability_zoom_counts.csv", index=False)
    bag.to_csv(args.output_dir / f"{NAME}_values.csv", index=False)
    pd.concat([pd.DataFrame({"variable": column, "bin_left": bins[:-1],
                            "bin_right": bins[1:], "count": counts[column]})
               for column, bins in BINS.items()]).to_csv(
                   args.output_dir / f"{NAME}_counts.csv", index=False)
    summary = pd.DataFrame({"n_pairs": values.count(), "mean": values.mean(),
                            "sd_ddof_0": values.std(ddof=0),
                            "minimum": values.min(), "maximum": values.max()})
    summary.to_csv(args.output_dir / f"{NAME}_summary.csv", index_label="variable")
    (args.output_dir / f"{NAME}_provenance.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(PIPELINE.parent)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "bins": {column: bins.tolist() for column, bins in BINS.items()}, "n_pairs": 30,
        "bin_convention": "Left-inclusive, right-exclusive; final bin includes its right edge.",
        "standard_deviation": "ddof=0, divisor 30; matches manual verification notebook.",
        "additional_zoom": {"selection": "prediction < 0.002", "n_pairs": len(zoom),
                            "n_pairs_outside_view": len(excluded), "bins": zoom_bins.tolist(),
                            "mean": float(zoom.prediction.mean()),
                            "sd_ddof_0": float(zoom.prediction.std(ddof=0))},
        "weighting": "One count per context-trigger-query pair; no response-count weighting.",
        "refitting_or_resampling": False,
    }, indent=2) + "\n")
    print(summary)
    print("Histogram counts:", {key: list(value) for key, value in counts.items()})
    print(f"Saved combined and separate histograms to {args.output_dir}")


if __name__ == "__main__":
    main()
