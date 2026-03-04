# Make 6 scatter plots:
#   x-axis = spearman_drop (from word_comparison_summary.csv)
#   y-axis = mean log likelihood per context (computed from each results file)
#   one point per context (15 contexts)
#
# Files expected in /mnt/data:
#   word_comparison_summary.csv           (must have: context, spearman_drop)
#   set_results_july2025.csv
#   ordering_results_july2025.csv
#   conjunction_results_july2025.csv
#   disjunction_results_july2025.csv
#   always_negate_results_july2025.csv
#   never_negate_results_july2025.csv
#
# Output:
#   ../figures/set_results_july2025_scatter.png
#   ../figures/ordering_results_july2025_scatter.png
#   ../figures/conjunction_results_july2025_scatter.png
#   ../figures/disjunction_results_july2025_scatter.png
#   ../figures/always_negate_results_july2025_scatter.png
#   ../figures/never_negate_results_july2025_scatter.png

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # for matplotlib.colormaps

BASE = "../results/"
OUTDIR = "../figures/"
os.makedirs(OUTDIR, exist_ok=True)

# Load per-context Spearman
summary_path = os.path.join(BASE, "word_comparison_summary.csv")
summary_df = pd.read_csv(summary_path, usecols=["context", "spearman_drop"]).dropna(subset=["context"])
summary_df = summary_df.drop_duplicates(subset=["context"])

# Define files
CORE_FILES = [
    ("set_results_july2025.csv", "Set"),
    ("ordering_results_july2025.csv", "Ordering"),
    ("conjunction_results_july2025.csv", "Conjunction"),
    ("disjunction_results_july2025.csv", "Disjunction"),
]
OTHER_FILES = [
    ("always_negate_results_july2025.csv", "Always Negate"),
    ("never_negate_results_july2025.csv", "Never Negate"),
]
ALL_FILES = CORE_FILES + OTHER_FILES

def mean_ll_by_context(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("context", as_index=False)["log_likelihood"]
              .mean()
              .rename(columns={"log_likelihood": "mean_log_likelihood"}))

def load_merged_for_file(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath, usecols=["context", "log_likelihood"])
    mean_ll = mean_ll_by_context(df)
    merged = pd.merge(mean_ll, summary_df, on="context", how="inner").dropna(subset=["spearman_drop", "mean_log_likelihood"])
    return merged.reset_index(drop=True)

# ---------- First pass: compute global X limits (all files) ----------
all_x = []
for fname, _ in ALL_FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        continue
    merged = load_merged_for_file(fpath)
    if not merged.empty:
        all_x.append(merged["spearman_drop"])
if all_x:
    x_series = pd.concat(all_x, ignore_index=True)
    x_min, x_max = x_series.min(), x_series.max()
    x_pad = (x_max - x_min) * 0.05 if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 0.05
    XLIM = (x_min - x_pad, x_max + x_pad)
else:
    XLIM = None

# ---------- Second pass: compute shared Y limits for CORE files only ----------
core_y = []
for fname, _ in CORE_FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        continue
    merged = load_merged_for_file(fpath)
    if not merged.empty:
        core_y.append(merged["mean_log_likelihood"])
if core_y:
    y_series = pd.concat(core_y, ignore_index=True)
    y_min, y_max = y_series.min(), y_series.max()
    y_pad = (y_max - y_min) * 0.05 if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 0.05
    CORE_YLIM = (y_min - y_pad, y_max + y_pad)
else:
    CORE_YLIM = None

# ---------- Consistent colors across ALL plots ----------
all_contexts = sorted(summary_df["context"].astype(str).unique().tolist())
cmap = matplotlib.colormaps["tab20"]
colors = {ctx: cmap(i % 20) for i, ctx in enumerate(all_contexts)}

# ---------- Simple label repulsion ----------
def repel_text_labels(ax, texts, max_iter=200, step=0.01, pad=2):
    fig = ax.figure
    fig.canvas.draw()
    transform = ax.transData
    inv_transform = ax.transData.inverted()
    for _ in range(max_iter):
        moved = False
        bboxes = [t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(
            1.0 + pad/100.0, 1.0 + pad/100.0) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bb1, bb2 = bboxes[i], bboxes[j]
                if bb1.overlaps(bb2):
                    dx = (bb1.x1 + bb1.x0)/2 - (bb2.x1 + bb2.x0)/2
                    dy = (bb1.y1 + bb1.y0)/2 - (bb2.y1 + bb2.y0)/2
                    if dx == 0 and dy == 0:
                        dx = 1.0
                    mag = (dx**2 + dy**2) ** 0.5
                    dx /= mag; dy /= mag
                    p1 = transform.transform(texts[i].get_position())
                    p2 = transform.transform(texts[j].get_position())
                    p1_new = p1 + np.array([dx, dy]) * step * 100
                    p2_new = p2 - np.array([dx, dy]) * step * 100
                    texts[i].set_position(inv_transform.transform(p1_new))
                    texts[j].set_position(inv_transform.transform(p2_new))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()

def plot_file(filename: str, title: str, dpi=600, figsize=(8, 6), use_core_ylim=False):
    fpath = os.path.join(BASE, filename)
    out = os.path.join(OUTDIR, f"{os.path.splitext(filename)[0]}_scatter.png")
    fig, ax = plt.subplots(figsize=figsize)

    if not os.path.exists(fpath):
        ax.set_title(f"{title}: file not found\n{filename}")
        ax.set_xlabel("Spearman (drop-missing)")
        ax.set_ylabel("Mean log likelihood")
        if XLIM: ax.set_xlim(XLIM)
        if use_core_ylim and CORE_YLIM: ax.set_ylim(CORE_YLIM)
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight", dpi=dpi); plt.close(fig)
        return out

    merged = load_merged_for_file(fpath)

    for _, row in merged.iterrows():
        ctx = str(row["context"])
        ax.scatter(row["spearman_drop"], row["mean_log_likelihood"],
                   s=40, color=colors.get(ctx, "black"),
                   edgecolors="black", linewidths=0.5)

    texts = []
    for _, row in merged.iterrows():
        t = ax.annotate(str(row["context"]),
                        (row["spearman_drop"], row["mean_log_likelihood"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        texts.append(t)

    repel_text_labels(ax, texts, max_iter=250, step=0.01, pad=2)

    ax.set_title(f"{title}: Mean LL vs Spearman (per context)")
    ax.set_xlabel("Spearman (drop-missing)")
    ax.set_ylabel("Mean log likelihood")
    if XLIM: ax.set_xlim(XLIM)
    if use_core_ylim and CORE_YLIM:
        ax.set_ylim(CORE_YLIM)
    else:
        # Independent Y with 5% padding for the non-core plots
        if not merged.empty:
            y_min, y_max = merged["mean_log_likelihood"].min(), merged["mean_log_likelihood"].max()
            y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.05
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out

saved = []
for fname, title in CORE_FILES:
    saved.append(plot_file(fname, title, use_core_ylim=True))
for fname, title in OTHER_FILES:
    saved.append(plot_file(fname, title, use_core_ylim=False))

print("Saved plots:", saved)
