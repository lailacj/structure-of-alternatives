"""Plot four prespecified models across datasets using saved corrected fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15"
MODELS = ["No linking structure", "X but not Y", "Ordering", "Set Top-K"]
LABELS = ["No linking", "X but not Y", "Ordering", "Set Top-K"]
COLORS = ["#737980", "#D55E00", "#0072B2", "#009E73"]
DATASETS = [
    ("van Tiel et al. (2016)", "van Tiel et al.\n(2016)"),
    ("Gotzner et al. (2018)", "Gotzner et al.\n(2018)"),
    ("Pankratz & van Tiel (2021)", "Pankratz &\nvan Tiel (2021)"),
    ("Ronai & Xiang (2022)", "Ronai & Xiang\n(2022)"),
    ("R&X ESI", "R&X (2024)\nESI"),
    ("R&X Eweak", "R&X (2024)\nEweak"),
    ("R&X Estrong", "R&X (2024)\nEstrong"),
    ("R&X Eonly", "R&X (2024)\nEonly"),
    ("R&X Eonlystrong", "R&X (2024)\nEonlystrong"),
    ("Novel Focus Alternative Study", "Novel focus"),
]
METRICS = {
    "log_score": ("log_scores", "Log score", "Mean log score (nats)"),
    "pearson_r": ("correlations", "Pearson correlation", "Pearson r"),
    "spearman_rho": ("spearman", "Spearman correlation", "Spearman ρ"),
}


def read_fits(results: Path) -> tuple[pd.DataFrame, list[Path]]:
    sources, records = [], []
    focus_path = results / "spearman/mean_within_context_spearman.csv"
    focus = pd.read_csv(focus_path)
    sources.append(focus_path)
    selectors = [("No linking structure", "direct"), ("X but not Y", "direct"),
                 ("ordering", "top_k"), ("set", "top_k")]
    for metric, (stem, _, _) in METRICS.items():
        path = results / "linking_structure_tables" / f"{stem}_by_dataset_and_linking_structure.csv"
        sources.append(path)
        table = pd.read_csv(path).set_index("Dataset")
        if not table.index.is_unique:
            raise ValueError(f"Duplicate dataset rows: {path}")
        for dataset, _ in DATASETS:
            for model, (structure, variant) in zip(MODELS, selectors):
                aggregation = "within_dataset"
                contexts = None
                if metric == "spearman_rho" and dataset == "Novel Focus Alternative Study":
                    rows = focus.loc[focus.measure.eq("negation") & focus.structure.eq(structure)
                                     & focus.variant.eq(variant)]
                    if len(rows) != 1:
                        raise ValueError(f"Expected one focus negation Spearman row for {model}")
                    row = rows.iloc[0]
                    value = float(row.mean_within_context_spearman)
                    contexts = int(row.valid_contexts)
                    if contexts != 16 or int(row.total_contexts) != 16:
                        raise ValueError("This figure expects 16 defined focus contexts for all four models")
                    aggregation = "equal_mean_of_16_within_context_negation_correlations"
                else:
                    value = float(table.loc[dataset, model])
                unavailable = model == "X but not Y" and dataset.startswith("R&X ")
                if unavailable != bool(np.isnan(value)):
                    raise ValueError(f"Unexpected coverage: {dataset}, {model}, {metric}")
                if not unavailable and not np.isfinite(value):
                    raise ValueError("Nonfinite fit")
                if metric != "log_score" and not unavailable and not -1 <= value <= 1:
                    raise ValueError("Correlation outside [-1, 1]")
                if metric == "log_score" and not unavailable and value > 0:
                    raise ValueError("Positive Bernoulli log score")
                records.append(dict(metric=metric, dataset=dataset, model=model, value=value,
                                    available=not unavailable, aggregation=aggregation,
                                    valid_focus_contexts=contexts))
    return pd.DataFrame(records), sources


def plot_metric(fits: pd.DataFrame, metric: str, output: Path) -> None:
    _, title, ylabel = METRICS[metric]
    fig, ax = plt.subplots(figsize=(16, 7.2))
    fig.subplots_adjust(left=.072, right=.985, bottom=.23, top=.76)
    fig.text(.072, .94, title, fontsize=22, fontweight="bold", color="#182733")
    direction = "Higher (closer to zero) is better" if metric == "log_score" else "Higher is better"
    fig.text(.072, .891, f"Corrected Qwen2-7B results  ·  {direction}", fontsize=12, color="#52616B")
    fig.legend([Patch(facecolor=c) for c in COLORS], LABELS,
               loc="upper left", bbox_to_anchor=(.065, .86), ncol=4,
               frameon=False, fontsize=12, handlelength=1.3, columnspacing=2.2)
    x = np.arange(len(DATASETS))
    width = .18
    subset = fits.loc[fits.metric.eq(metric)]
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        values = subset.loc[subset.model.eq(model)].set_index("dataset").loc[[d for d, _ in DATASETS], "value"].to_numpy()
        centers = x + (i - 1.5) * width
        valid = np.isfinite(values)
        ax.bar(centers[valid], values[valid], width=width*.92, color=color,
               edgecolor="white", linewidth=.5, zorder=3)
        for center in centers[~valid]:
            ax.annotate("N/A", (center, 0), xytext=(0, -9 if metric == "log_score" else 7),
                        textcoords="offset points", ha="center",
                        va="top" if metric == "log_score" else "bottom",
                        rotation=90, fontsize=8, color="#64717A")
    ax.axhline(0, color="#354550", linewidth=1, zorder=4)
    ax.set_xlim(-.6, 9.6)
    if metric == "log_score":
        lower = np.floor(subset.value.min()) - .35
        ax.set_ylim(lower, 0)
        ax.yaxis.set_major_locator(MultipleLocator(1))
    else:
        ax.set_ylim(-.35, .70)
        ax.yaxis.set_major_locator(MultipleLocator(.2))
    labels = [label + ("*" if metric == "spearman_rho" and i == 9 else "")
              for i, (_, label) in enumerate(DATASETS)]
    ax.set_xticks(x, labels, fontsize=10.5, linespacing=1.4)
    ax.tick_params(axis="x", length=0, pad=14)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=11)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=12)
    ax.set_xlabel("Dataset", fontsize=12, labelpad=13)
    ax.grid(axis="y", color="#DCE2E6", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD3D9")
    fig.text(.072, .075, "N/A: X but not Y is unavailable for the five R&X (2024) conditions. Bars show point estimates.",
             fontsize=10, color="#52616B")
    if metric == "spearman_rho":
        note = "*Novel focus: mean negation ρ across 16 contexts (30 pairs per context). Other datasets: within-dataset ρ."
    else:
        note = "Novel focus uses all 480 pairs. Hu scales are evaluated after template aggregation; R&X conditions use 60 items each."
    fig.text(.072, .042, note, fontsize=10, color="#52616B")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"four_models_{metric}.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir / "advisor_summary/four_model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    fits, sources = read_fits(args.results_dir)
    fits.to_csv(output / "plotted_fit_values.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    for metric in METRICS:
        plot_metric(fits, metric, output)
    provenance = {"results_directory": str(args.results_dir.resolve()),
                  "models": MODELS, "missing_values": "not plotted as zero",
                  "source_files": [{"path": str(path.resolve()),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                   for path in sources]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "README.md").write_text("""# Four-model fit comparisons

Three grouped bar charts show No linking, X but not Y, Ordering, and Set Top-K
across the ten datasets. Each figure is provided as PNG, PDF and editable SVG.
Colors and model order are identical throughout. Higher values mean better fit;
negative log scores extend down from zero. Correlation axes share the same range.

Figures use the completed corrected single-space results. The bars are descriptive
point estimates, with no uncertainty interval or claim of statistical superiority.
Missing X-but-not-Y values for the five R&X conditions are labeled N/A.

Log scores and Pearson correlations are copied from the aligned dataset tables.
Hu and R&X Spearman values use the saved within-dataset/condition coefficients.
Novel-focus Spearman is the equal-weight mean of the 16 within-context negation
correlations (30 ordered pairs each), as in the current viewer. It is neither
word-ranking Spearman nor a pooled correlation over the 480 pairs.

`plotted_fit_values.csv` records every bar or unavailable slot, its aggregation,
and valid focus-context count. `provenance.json` records input hashes.

Regenerate from the repository root:

```bash
python focus_alt_exp_pipeline/code/plot_four_model_fit_comparison.py
```
""", encoding="utf-8")
    print(f"Saved three figures in PNG/PDF/SVG, values and provenance to {output}")


if __name__ == "__main__":
    main()
