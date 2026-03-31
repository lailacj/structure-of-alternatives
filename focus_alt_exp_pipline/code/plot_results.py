"""Plot per-context average log likelihood within one next-word model.

This script is intentionally narrow in scope. For one model's result folder
(for example ``results/cloze_probability``), it makes a plot with:

- x-axis: alternative structure (`set`, `ordering`, `conjunction`, `disjunction`)
- y-axis: average log likelihood
- one dot per context for each alternative structure
- one black dot per alternative structure showing the mean across contexts

It also writes the summary CSVs that feed the plot.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


FILE_PATTERN = re.compile(r"^(set|ordering|conjunction|disjunction)_results_(.+)\.csv$")
DEFAULT_STRUCTURE_ORDER = ["set", "ordering", "conjunction", "disjunction"]
STRUCTURE_LABELS = {
    "set": "Set",
    "ordering": "Ordering",
    "conjunction": "Conjunction",
    "disjunction": "Disjunction",
}


def _parse_csv_list(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _safe_stem(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return safe.strip("_")


def _pretty_model_name(raw_name: str) -> str:
    acronym_map = {
        "frequency": "Frequency",
        "qwen": "Qwen",
        "cloze": "Cloze",
        "uniform": "Uniform",
    }
    parts = raw_name.replace("-", "_").split("_")
    return " ".join(acronym_map.get(part.lower(), part.capitalize()) for part in parts)


def _collect_results(
    results_dir: Path,
    *,
    structures: Iterable[str],
    model_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    structure_set = set(structures)
    frames: list[pd.DataFrame] = []
    found_models: set[str] = set()

    for path in sorted(results_dir.glob("*_results_*.csv")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        structure, raw_model_name = match.groups()
        if structure not in structure_set:
            continue
        if model_name is not None and raw_model_name != model_name:
            continue

        df = pd.read_csv(path)
        required = {"context", "log_likelihood"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        frame = df.copy()
        frame["AlternativeStructure"] = structure
        frame["NextWordPredictionModelRaw"] = raw_model_name
        frame["SourceFile"] = path.name
        frames.append(frame)
        found_models.add(raw_model_name)

    if not frames:
        raise FileNotFoundError(
            f"No matching result CSVs found in {results_dir} for structures {sorted(structure_set)}"
        )

    if model_name is None and len(found_models) > 1:
        raise ValueError(
            "Multiple next-word models were found in the same results directory. "
            "Pass --model to choose one explicitly."
        )

    resolved_model = model_name if model_name is not None else next(iter(found_models))
    return pd.concat(frames, ignore_index=True), resolved_model


def _summarize_by_context(
    all_results: pd.DataFrame,
    *,
    structure_order: List[str],
) -> pd.DataFrame:
    summary = (
        all_results.groupby(
            ["NextWordPredictionModelRaw", "AlternativeStructure", "context"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            observed_rows=("log_likelihood", "size"),
            observed_boundaries=("set_boundary", "nunique"),
            average_log_likelihood=("log_likelihood", "mean"),
        )
    )
    summary["AlternativeStructure"] = pd.Categorical(
        summary["AlternativeStructure"],
        categories=structure_order,
        ordered=True,
    )
    return summary.sort_values(
        ["AlternativeStructure", "context"],
        ignore_index=True,
    )


def _summarize_structure_means(
    context_summary: pd.DataFrame,
    *,
    structure_order: List[str],
) -> pd.DataFrame:
    summary = (
        context_summary.groupby(
            ["NextWordPredictionModelRaw", "AlternativeStructure"],
            as_index=False,
            sort=False,
            observed=False,
        )
        .agg(
            context_count=("context", "nunique"),
            mean_log_likelihood=("average_log_likelihood", "mean"),
        )
    )
    summary["AlternativeStructure"] = pd.Categorical(
        summary["AlternativeStructure"],
        categories=structure_order,
        ordered=True,
    )
    return summary.sort_values(["AlternativeStructure"], ignore_index=True)


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _make_plot(
    context_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    *,
    structure_order: List[str],
    model_display_name: str,
    output_path: Path,
    title: str | None,
    point_alpha: float,
    point_size: float,
    mean_point_size: float,
    jitter: float,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    x_positions = {structure: idx for idx, structure in enumerate(structure_order)}

    for structure in structure_order:
        structure_rows = context_summary[
            context_summary["AlternativeStructure"].astype(str) == structure
        ].copy()
        if structure_rows.empty:
            continue

        x0 = x_positions[structure]
        offsets = rng.uniform(-jitter, jitter, size=len(structure_rows)) if jitter > 0 else 0.0
        ax.scatter(
            np.full(len(structure_rows), x0, dtype=float) + offsets,
            structure_rows["average_log_likelihood"],
            s=point_size,
            color="#4C78A8",
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        mean_rows = mean_summary[mean_summary["AlternativeStructure"].astype(str) == structure]
        if not mean_rows.empty:
            ax.scatter(
                [x0],
                [float(mean_rows.iloc[0]["mean_log_likelihood"])],
                s=mean_point_size,
                color="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )

    ax.set_xlim(-0.5, len(structure_order) - 0.5)
    ax.set_xticks(
        list(x_positions.values()),
        [STRUCTURE_LABELS.get(structure, structure.title()) for structure in structure_order],
    )
    ax.set_xlabel("Alternative Structure")
    ax.set_ylabel("Average Log Likelihood")
    ax.set_title(
        title if title is not None else f"{model_display_name}: Average Log Likelihood by Structure"
    )

    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_results_dir = root_dir / "focus_alt_exp_pipline" / "results" / "cloze_probability"

    parser = argparse.ArgumentParser(
        description="Plot per-context average log likelihood within one next-word model"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help="Directory containing one model's raw *_results_<model>.csv files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional raw model name such as cloze or frequency. Usually inferred from filenames.",
    )
    parser.add_argument(
        "--structures",
        type=str,
        default=",".join(DEFAULT_STRUCTURE_ORDER),
        help="Comma-separated alternative structures in plotting order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plot outputs (default: <results-dir>/plots).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument("--point-alpha", type=float, default=0.75)
    parser.add_argument("--point-size", type=float, default=46.0)
    parser.add_argument("--mean-point-size", type=float, default=95.0)
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.10,
        help="Horizontal jitter for context dots.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure_order = _parse_csv_list(args.structures)
    if not structure_order:
        raise ValueError("--structures produced an empty list")

    model_name = args.model.strip() or None
    out_dir = args.output_dir if args.output_dir is not None else args.results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results, resolved_model_name = _collect_results(
        args.results_dir,
        structures=structure_order,
        model_name=model_name,
    )
    context_summary = _summarize_by_context(
        all_results,
        structure_order=structure_order,
    )
    mean_summary = _summarize_structure_means(
        context_summary,
        structure_order=structure_order,
    )

    model_stem = _safe_stem(resolved_model_name)
    model_display_name = _pretty_model_name(resolved_model_name)
    context_csv = out_dir / f"average_log_likelihood_by_context_and_structure__{model_stem}.csv"
    mean_csv = out_dir / f"mean_log_likelihood_by_structure__{model_stem}.csv"
    plot_path = out_dir / f"log_likelihood_by_structure_with_context_dots__{model_stem}.png"

    context_summary.to_csv(context_csv, index=False)
    mean_summary.to_csv(mean_csv, index=False)
    _make_plot(
        context_summary,
        mean_summary,
        structure_order=structure_order,
        model_display_name=model_display_name,
        output_path=plot_path,
        title=args.title,
        point_alpha=args.point_alpha,
        point_size=args.point_size,
        mean_point_size=args.mean_point_size,
        jitter=args.jitter,
        seed=args.seed,
    )

    print("Saved:", context_csv)
    print("Saved:", mean_csv)
    print("Saved:", plot_path)


if __name__ == "__main__":
    main()
