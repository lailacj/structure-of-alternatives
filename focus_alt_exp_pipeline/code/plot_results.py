"""Plot per-context average log likelihood within one next-word model.

This script is intentionally narrow in scope. For one model's result folder
(for example ``results/cloze_probability``), it makes a plot with:

- x-axis: alternative structure (`set`, `ordering`, `conjunction`, `disjunction`)
- y-axis: average log likelihood
- one color-coded dot per context for each alternative structure
- one black dot per alternative structure showing the mean across contexts
- a legend mapping colors to contexts
- one correlation scatter plot per alternative structure comparing model and human
  negation probabilities by trial type

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


def _resolve_human_data_default() -> Path:
    return Path(__file__).resolve().parents[1] / "human_exp_data" / "sca_dataframe.csv"


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


def _build_context_color_map(contexts: list[str], *, cmap_name: str = "tab20") -> dict[str, object]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    return {
        context: cmap(idx % cmap.N)
        for idx, context in enumerate(sorted(contexts))
    }


def _legend_handles_for_contexts(contexts: list[str], context_colors: dict[str, object]):
    import matplotlib.pyplot as plt

    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=context_colors[context],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7.5,
            label=context,
        )
        for context in sorted(contexts)
    ]


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
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x_positions = {structure: idx for idx, structure in enumerate(structure_order)}
    contexts = sorted(context_summary["context"].astype(str).unique().tolist())
    context_colors = _build_context_color_map(contexts)

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
            c=[context_colors[str(context)] for context in structure_rows["context"]],
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
    legend_handles = _legend_handles_for_contexts(contexts, context_colors)
    ax.legend(
        handles=legend_handles,
        title="Context",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    _style_axes(ax)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _load_human_trial_probabilities(human_data_path: Path) -> pd.DataFrame:
    human_df = pd.read_csv(human_data_path)

    context_col = "story" if "story" in human_df.columns else "context"
    query_col = "cleaned_query" if "cleaned_query" in human_df.columns else "query"
    trigger_col = "cleaned_trigger" if "cleaned_trigger" in human_df.columns else "trigger"
    required = {context_col, query_col, trigger_col, "neg"}
    missing = required.difference(human_df.columns)
    if missing:
        raise ValueError(
            f"{human_data_path} is missing required columns: {sorted(missing)}"
        )

    prepared = human_df[[context_col, trigger_col, query_col, "neg"]].copy()
    prepared.columns = ["context", "trigger", "query", "neg"]
    prepared["context"] = prepared["context"].astype(str).str.strip()
    prepared["trigger"] = prepared["trigger"].astype(str).str.strip()
    prepared["query"] = prepared["query"].astype(str).str.strip()
    prepared["neg"] = pd.to_numeric(prepared["neg"], errors="raise")

    summary = (
        prepared.groupby(
            ["context", "trigger", "query"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            human_negation_probability=("neg", "mean"),
            human_trial_count=("neg", "size"),
        )
    )
    return summary.sort_values(["context", "trigger", "query"], ignore_index=True)


def _summarize_model_trial_probabilities(
    all_results: pd.DataFrame,
    *,
    structure_order: list[str],
) -> pd.DataFrame:
    summary = (
        all_results.groupby(
            [
                "NextWordPredictionModelRaw",
                "AlternativeStructure",
                "context",
                "trigger",
                "query",
            ],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            model_negation_probability=("negation_probability", "mean"),
            observed_rows=("negation_probability", "size"),
            observed_boundaries=("set_boundary", "nunique"),
        )
    )
    summary["AlternativeStructure"] = pd.Categorical(
        summary["AlternativeStructure"],
        categories=structure_order,
        ordered=True,
    )
    return summary.sort_values(
        ["AlternativeStructure", "context", "trigger", "query"],
        ignore_index=True,
    )


def _build_correlation_summary(
    all_results: pd.DataFrame,
    *,
    structure_order: list[str],
    human_data_path: Path,
) -> pd.DataFrame:
    model_summary = _summarize_model_trial_probabilities(
        all_results,
        structure_order=structure_order,
    )
    human_summary = _load_human_trial_probabilities(human_data_path)
    merged = model_summary.merge(
        human_summary,
        on=["context", "trigger", "query"],
        how="inner",
    )
    return merged.sort_values(
        ["AlternativeStructure", "context", "trigger", "query"],
        ignore_index=True,
    )


def _pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    if len(x_values) < 2:
        return float("nan")
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _make_correlation_plots(
    correlation_summary: pd.DataFrame,
    *,
    structure_order: list[str],
    model_display_name: str,
    model_stem: str,
    output_dir: Path,
    point_alpha: float,
    point_size: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    contexts = sorted(correlation_summary["context"].astype(str).unique().tolist())
    context_colors = _build_context_color_map(contexts)
    legend_handles = _legend_handles_for_contexts(contexts, context_colors)
    saved_paths: list[Path] = []

    for structure in structure_order:
        structure_rows = correlation_summary[
            correlation_summary["AlternativeStructure"].astype(str) == structure
        ].copy()
        if structure_rows.empty:
            continue

        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        ax.scatter(
            structure_rows["model_negation_probability"],
            structure_rows["human_negation_probability"],
            s=point_size,
            c=[context_colors[str(context)] for context in structure_rows["context"]],
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, linewidth=1.0, zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Model Negation Probability")
        ax.set_ylabel("Human Negation Probability")

        corr_value = _pearson_correlation(
            structure_rows["model_negation_probability"],
            structure_rows["human_negation_probability"],
        )
        structure_label = STRUCTURE_LABELS.get(structure, structure.title())
        title = f"{model_display_name}: {structure_label} Negation Probability Correlation"
        if not np.isnan(corr_value):
            title = f"{title}\nPearson r = {corr_value:.3f}"
        ax.set_title(title)

        ax.legend(
            handles=legend_handles,
            title="Context",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            borderaxespad=0.0,
        )
        _style_axes(ax)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

        plot_path = output_dir / (
            f"negation_probability_correlation__{_safe_stem(structure)}__{model_stem}.png"
        )
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_results_dir = root_dir / "focus_alt_exp_pipeline" / "results" / "cloze_probability"

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
    parser.add_argument(
        "--human-data",
        type=Path,
        default=_resolve_human_data_default(),
        help="Human participant-level CSV used to compute human negation probabilities.",
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
    correlation_csv = out_dir / f"negation_probability_correlation_points__{model_stem}.csv"

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
    correlation_summary = _build_correlation_summary(
        all_results,
        structure_order=structure_order,
        human_data_path=args.human_data,
    )
    correlation_summary.to_csv(correlation_csv, index=False)
    correlation_plot_paths = _make_correlation_plots(
        correlation_summary,
        structure_order=structure_order,
        model_display_name=model_display_name,
        model_stem=model_stem,
        output_dir=out_dir,
        point_alpha=args.point_alpha,
        point_size=args.point_size,
    )

    print("Saved:", context_csv)
    print("Saved:", mean_csv)
    print("Saved:", plot_path)
    print("Saved:", correlation_csv)
    for saved_path in correlation_plot_paths:
        print("Saved:", saved_path)


if __name__ == "__main__":
    main()
