"""Plot saved bag probabilities for No linking and Set Top-K without rescoring.

Requires numpy, pandas and matplotlib. Outputs PNG, PDF, SVG, plotted data,
caption and input provenance to <repository>/output/pdf by default.
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
NAME = "bag_no_linking_set_topk_scatter"
MODELS = [("No linking structure", "No linking"), ("Set Top-K", "Set Top-K")]
CAPTION = (
    "Model exclusion probabilities versus observed human exclusion rates in the bag "
    "context. Each panel contains the same 30 ordered trigger-query pairs, using "
    "saved corrected predictions. Both axes span 0 to 1 on the same linear scale. "
    "The dashed equality line marks p = y; points above it have model probabilities "
    "below observed human rates. Marker area is proportional to the number of pairs "
    "at identical coordinates, using the same size scale in both panels. Coordinates "
    "are not jittered. Means weight each of the 30 pairs equally. Human rates are "
    "sample estimates; equality here describes agreement with those observed rates."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parent / "output/pdf")
    args = parser.parse_args()
    data = pd.read_csv(SOURCE)
    frames = []
    for model, _ in MODELS:
        frame = data.loc[data.context_id.eq("bag") & data.model.eq(model), [
            "model", "item_id", "context_id", "trigger", "query", "prediction", "human_rate"
        ]].sort_values("item_id")
        assert len(frame) == frame.item_id.nunique() == 30
        assert np.isfinite(frame[["prediction", "human_rate"]].to_numpy()).all()
        assert frame[["prediction", "human_rate"]].ge(0).all().all()
        assert frame[["prediction", "human_rate"]].le(1).all().all()
        frames.append(frame)
    assert frames[0].item_id.tolist() == frames[1].item_id.tolist()
    np.testing.assert_array_equal(frames[0].human_rate, frames[1].human_rate)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.25, top=.81, wspace=.15)
    fig.suptitle("Model probabilities and human exclusion rates", y=.975,
                 fontsize=19, fontweight="bold")
    fig.text(.5, .92, "Bag context | 30 trigger-query pairs per model",
             ha="center", fontsize=13, color="#53616E")
    for ax, (_, label), frame in zip(axes, MODELS, frames):
        points = frame.groupby(["prediction", "human_rate"]).size().reset_index(name="n_pairs")
        ax.set_axisbelow(True)
        ax.grid(color="#E6EAEE", linewidth=.8)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7C8790", linewidth=1.4)
        ax.scatter(points.prediction, points.human_rate, s=48 * points.n_pairs,
                   color="#176B92", edgecolor="white", linewidth=.65, alpha=.8,
                   zorder=3, clip_on=False)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
               xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6),
               xlabel="Model exclusion probability")
        ax.set_title(label, fontsize=16, fontweight="bold", pad=12)
        ax.text(.5, -.23,
                f"Mean model: {frame.prediction.mean():.4f}  |  Mean human: {frame.human_rate.mean():.4f}",
                transform=ax.transAxes, ha="center", fontsize=11, color="#374754")
    axes[0].set_ylabel("Human exclusion rate")
    fig.text(.5, .07, "Dashed line: model probability = observed human rate",
             ha="center", fontsize=11, color="#53616E")
    fig.text(.5, .025, "Larger markers represent pairs at identical coordinates; no jitter.",
             ha="center", fontsize=10, color="#53616E")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.output_dir / f"{NAME}.{ext}", dpi=250)
    plt.close(fig)
    pd.concat(frames).to_csv(args.output_dir / f"{NAME}.csv", index=False)
    (args.output_dir / f"{NAME}_caption.txt").write_text(CAPTION + "\n")
    (args.output_dir / f"{NAME}_provenance.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(PIPELINE.parent)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "context": "bag", "models": [model for model, _ in MODELS],
        "n_pairs_per_model": 30, "refitting_or_resampling": False,
        "caption": CAPTION,
    }, indent=2) + "\n")
    print(f"Saved {args.output_dir / NAME} (.png, .pdf, .svg, .csv, caption, provenance)")


if __name__ == "__main__":
    main()
