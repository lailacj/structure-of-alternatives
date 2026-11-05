#!/usr/bin/env python3
"""Plot the validated 16-context table without computing inferential statistics.

Requires NumPy and Matplotlib. Reads only the saved CSV and its provenance;
does not access model weights, candidate arrays, or training/scoring code.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


HERE = Path(__file__).resolve().parent
# Offsets only position labels; point coordinates always come from the CSV.
LABEL_OFFSETS = {
    "mask": (8, -15), "bag": (8, 10), "fridge": (8, 9),
    "hot": (-9, -15), "mall": (8, 10), "cold": (-9, 10),
    "throw": (9, -12), "science": (-9, -10), "restaurant": (8, 10),
    "transport": (-9, 10), "cut": (8, -15), "salad": (8, 10),
    "gym": (8, 10), "bakery": (-9, -15), "meat": (8, -15),
    "beach": (8, 10),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=HERE / "entropy_ordering_advantage.csv")
    parser.add_argument("--analysis-provenance", type=Path, default=HERE / "provenance_validation.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "scatter_plot")
    args = parser.parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    targets = {extension: output_dir / f"entropy_ordering_advantage_scatter.{extension}"
               for extension in ("png", "pdf", "svg")}
    provenance_path = output_dir / "plot_provenance.json"
    for path in (*targets.values(), provenance_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    source_provenance = json.loads(args.analysis_provenance.read_text())
    if source_provenance["validation_passed"] is not True or digest(input_csv) != source_provenance["output_csv"]["sha256"]:
        raise ValueError("Input CSV does not match the previously validated analysis")
    with input_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    contexts = [row["context"] for row in rows]
    x = np.array([float(row["entropy_bits"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["ordering_advantage"]) for row in rows], dtype=np.float64)
    if len(rows) != 16 or len(set(contexts)) != 16 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected exactly 16 unique contexts with finite coordinates")
    if not all(int(row["n_candidates"]) == 121301 and int(row["n_trigger_query_pairs"]) == 30 for row in rows):
        raise ValueError("Unexpected candidate/pair counts")
    for row in rows:
        if float(row["ordering_advantage"]) != float(row["ordering_log_score"]) - float(row["set_log_score"]):
            raise ValueError(f"Advantage arithmetic mismatch: {row['context']}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.edgecolor": "#8A949D",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#46525D", "ytick.color": "#46525D",
        "text.color": "#23313D", "axes.labelcolor": "#23313D",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(11.5, 7.3), dpi=150)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.17, top=0.82)
    fig.text(0.12, 0.94, "Entropy and Ordering’s advantage over Set", fontsize=18, weight="bold", va="top")
    fig.text(0.12, 0.885, "16 focus contexts  ·  corrected neutral distributions  ·  frozen K = 20", fontsize=11, color="#53616D")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6EBEF", linewidth=0.8)
    ax.axhline(0, color="#7A8791", linewidth=1, linestyle=(0, (4, 4)), zorder=2)
    points = ax.scatter(x, y, s=70, color="#166F7D", edgecolors="white", linewidths=0.9, zorder=4)
    labels = []
    for context, entropy, advantage in zip(contexts, x, y):
        dx, dy = LABEL_OFFSETS.get(context, (8, 10))
        labels.append(ax.annotate(
            context.upper(), (entropy, advantage), xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx >= 0 else "right", va="center", fontsize=10,
            annotation_clip=False, zorder=5,
        ))
    ax.set_xlabel("Candidate-distribution entropy (bits)", labelpad=13)
    ax.set_ylabel("Ordering advantage over Set\n(summed log score: Ordering − Set)", labelpad=12)
    ax.set_xlim(float(x.min()) - 0.45, float(x.max()) + 0.6)
    ax.set_ylim(min(-45.0, float(y.min()) - 17), float(y.max()) + 22)
    ax.set_xticks(np.arange(2, 8, 1))
    ax.set_yticks(np.arange(-25, 151, 25))
    ax.tick_params(length=4, pad=7)
    fig.text(0.12, 0.055, "Positive advantage: Ordering fits better.  Negative advantage: Set fits better.", fontsize=10, color="#53616D")
    fig.text(0.12, 0.025, "Each point uses all 121,301 candidates and the same 30 trigger–query pairs per context.", fontsize=10, color="#53616D")

    # Verify every saved point and label, and reject overlap/clipping before export.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Text.get_window_extent(label, renderer=renderer) for label in labels]
    overlaps = [(contexts[i], contexts[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                if boxes[i].overlaps(boxes[j])]
    if overlaps:
        raise ValueError(f"Context labels overlap; adjust offsets: {overlaps}")
    if any(box.x0 < fig.bbox.x0 or box.x1 > fig.bbox.x1 or box.y0 < fig.bbox.y0 or box.y1 > fig.bbox.y1 for box in boxes):
        raise ValueError("A context label extends outside the figure")
    np.testing.assert_array_equal(np.asarray(points.get_offsets()), np.column_stack((x, y)))
    if {label.get_text() for label in labels} != {context.upper() for context in contexts}:
        raise ValueError("Plot labels do not match all 16 contexts")

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, path in targets.items():
        with path.open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=300, facecolor="white")
    plt.close(fig)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "input_csv": {"path": str(input_csv), "sha256": digest(input_csv)},
        "analysis_provenance": {"path": str(args.analysis_provenance.resolve()), "sha256": digest(args.analysis_provenance)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "axes": {"horizontal": "entropy_bits", "vertical": "ordering_advantage", "reference_line": 0},
        "validation": {"n_contexts": 16, "all_contexts_labeled": True, "coordinates_match_csv": True,
                       "label_overlap_count": 0, "clipped_label_count": 0,
                       "distinct_entropy_values": int(np.unique(x).size),
                       "distinct_advantage_values": int(np.unique(y).size)},
        "outputs": [{"path": str(path), "sha256": digest(path)} for path in targets.values()],
        "statistical_tests_or_model_fitting_run": False,
    }
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Validated 16 labeled points with no label overlaps or clipping. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
