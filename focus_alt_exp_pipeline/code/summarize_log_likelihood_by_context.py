"""Summarize average log likelihood by context and alternative structure.

This script reads raw focus_alt_exp result CSVs such as
``ordering_results_cloze.csv`` and ``set_results_frequency.csv`` and produces
per-context summary tables for the requested models and structures.
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
DEFAULT_STRUCTURE_FILTER = ["set", "ordering", "conjunction", "disjunction"]
DEFAULT_MODEL_FILTER = ["cloze", "frequency"]
STRUCTURE_LABELS = {
    "set": "Set",
    "ordering": "Ordering",
    "conjunction": "Conjunction",
    "disjunction": "Disjunction",
}


def _parse_csv_list(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _pretty_model_name(raw_name: str) -> str:
    acronym_map = {
        "frequency": "Frequency",
        "qwen": "Qwen",
        "cloze": "Cloze Probability",
        "uniform": "Uniform",
    }
    parts = raw_name.replace("-", "_").split("_")
    return " ".join(acronym_map.get(part.lower(), part.capitalize()) for part in parts)


def _safe_stem(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return safe.strip("_")


def _resolve_human_data_default() -> Path:
    return Path(__file__).resolve().parents[1] / "human_exp_data" / "sca_dataframe.csv"


def _collect_results(
    results_dir: Path,
    structures: Iterable[str],
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    structure_set = set(structures)
    model_set = set(model_names) if model_names else None
    frames = []

    for path in sorted(results_dir.rglob("*_results_*.csv")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        structure, model_raw = match.groups()
        if structure not in structure_set:
            continue
        if model_set is not None and model_raw not in model_set:
            continue

        df = pd.read_csv(path)
        required_cols = {"context", "set_boundary", "log_likelihood"}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            missing_str = ", ".join(sorted(missing_cols))
            raise ValueError(f"Missing required columns in {path}: {missing_str}")

        frame = df.copy()
        frame["AlternativeStructure"] = structure
        frame["NextWordPredictionModelRaw"] = model_raw
        frame["NextWordPredictionModel"] = _pretty_model_name(model_raw)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No result files found in {results_dir} matching *_results_<model>.csv "
            f"for structures {sorted(structure_set)}"
        )

    return pd.concat(frames, ignore_index=True)


def _align_frequency_to_cloze(
    all_results: pd.DataFrame,
    *,
    reference_model: str = "cloze",
    target_model: str = "frequency",
) -> pd.DataFrame:
    required_cols = {"context", "trigger", "query", "NextWordPredictionModelRaw"}
    if not required_cols.issubset(all_results.columns):
        return all_results

    model_names = set(all_results["NextWordPredictionModelRaw"].astype(str).unique())
    if reference_model not in model_names or target_model not in model_names:
        return all_results

    key_cols = ["context", "trigger", "query"]
    reference_keys = (
        all_results.loc[all_results["NextWordPredictionModelRaw"] == reference_model, key_cols]
        .drop_duplicates()
        .assign(_keep=1)
    )

    target_rows = all_results.loc[all_results["NextWordPredictionModelRaw"] == target_model].copy()
    other_rows = all_results.loc[all_results["NextWordPredictionModelRaw"] != target_model].copy()
    aligned_target = target_rows.merge(reference_keys, on=key_cols, how="inner").drop(columns="_keep")

    return pd.concat([other_rows, aligned_target], ignore_index=True)


def _summarize_results_by_context(all_results: pd.DataFrame) -> pd.DataFrame:
    grouped = all_results.groupby(
        ["NextWordPredictionModelRaw", "NextWordPredictionModel", "AlternativeStructure", "context"],
        as_index=False,
        sort=False,
        dropna=False,
    )

    summary = grouped.agg(
        observed_rows=("log_likelihood", "size"),
        observed_boundaries=("set_boundary", "nunique"),
        average_log_likelihood=("log_likelihood", "mean"),
        median_log_likelihood=("log_likelihood", "median"),
        std_log_likelihood=("log_likelihood", "std"),
        min_log_likelihood=("log_likelihood", "min"),
        max_log_likelihood=("log_likelihood", "max"),
    )
    summary["std_log_likelihood"] = summary["std_log_likelihood"].fillna(0.0)
    return summary


def _summarize_structure_means(
    context_summary: pd.DataFrame,
    *,
    structure_order: list[str],
) -> pd.DataFrame:
    # With pandas 2.2.x, grouping a categorical column here with observed=False
    # can expand to unobserved cartesian combinations and then fail when the
    # group labels are inserted back into the aggregated frame.
    summary = (
        context_summary.groupby(
            ["NextWordPredictionModelRaw", "NextWordPredictionModel", "AlternativeStructure"],
            as_index=False,
            sort=False,
            observed=True,
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
    return summary


def _summarize_model_means(structure_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        structure_summary.groupby(
            ["NextWordPredictionModelRaw", "NextWordPredictionModel"],
            as_index=False,
            sort=False,
        )
        .agg(
            structure_count=("AlternativeStructure", "nunique"),
            overall_mean_log_likelihood=("mean_log_likelihood", "mean"),
        )
    )


def _load_human_trial_probabilities(human_data_path: Path) -> pd.DataFrame:
    human_df = pd.read_csv(human_data_path)

    context_col = "story" if "story" in human_df.columns else "context"
    query_col = "cleaned_query" if "cleaned_query" in human_df.columns else "query"
    trigger_col = "cleaned_trigger" if "cleaned_trigger" in human_df.columns else "trigger"
    required = {context_col, query_col, trigger_col, "neg"}
    missing = required.difference(human_df.columns)
    if missing:
        raise ValueError(f"{human_data_path} is missing required columns: {sorted(missing)}")

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
                "NextWordPredictionModel",
                "AlternativeStructure",
                "context",
                "trigger",
                "query",
            ],
            as_index=False,
            sort=False,
            dropna=False,
            observed=True,
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
        ["NextWordPredictionModel", "AlternativeStructure", "context", "trigger", "query"],
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
        ["NextWordPredictionModel", "AlternativeStructure", "context", "trigger", "query"],
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


def _summarize_structure_correlations(
    correlation_summary: pd.DataFrame,
    *,
    structure_order: list[str],
) -> pd.DataFrame:
    rows = []
    grouped = correlation_summary.groupby(
        ["NextWordPredictionModelRaw", "NextWordPredictionModel", "AlternativeStructure"],
        sort=False,
        dropna=False,
        observed=True,
    )
    for (model_raw, model_display, structure), group in grouped:
        rows.append(
            {
                "NextWordPredictionModelRaw": model_raw,
                "NextWordPredictionModel": model_display,
                "AlternativeStructure": structure,
                "trial_type_count": len(group),
                "context_count": group["context"].nunique(),
                "pearson_correlation": _pearson_correlation(
                    group["model_negation_probability"],
                    group["human_negation_probability"],
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No overlapping model and human trial types were found for correlation summary.")

    summary["AlternativeStructure"] = pd.Categorical(
        summary["AlternativeStructure"],
        categories=structure_order,
        ordered=True,
    )
    return summary.sort_values(
        ["NextWordPredictionModel", "AlternativeStructure"],
        ignore_index=True,
    )


def _summarize_model_correlation_means(
    structure_correlation_summary: pd.DataFrame,
) -> pd.DataFrame:
    return (
        structure_correlation_summary.groupby(
            ["NextWordPredictionModelRaw", "NextWordPredictionModel"],
            as_index=False,
            sort=False,
        )
        .agg(
            structure_count=("AlternativeStructure", "nunique"),
            mean_pearson_correlation=("pearson_correlation", "mean"),
        )
    )


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _build_structure_color_map(structure_order: list[str]) -> dict[str, object]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return {
        structure: cmap(idx % cmap.N)
        for idx, structure in enumerate(structure_order)
    }


def _legend_handles_for_structures(
    structure_order: list[str],
    structure_colors: dict[str, object],
):
    import matplotlib.pyplot as plt

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=structure_colors[structure],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8.0,
            label=STRUCTURE_LABELS.get(structure, structure.title()),
        )
        for structure in structure_order
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9.0,
            label="Mean Across Structures",
        )
    )
    return handles


def _make_across_model_plot(
    structure_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    *,
    model_order: list[str],
    structure_order: list[str],
    output_path: Path,
    title: str | None,
    point_size: float,
    mean_point_size: float,
    structure_offset: float,
    structure_value_col: str,
    model_value_col: str,
    y_label: str,
    default_title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    present_models = structure_summary["NextWordPredictionModelRaw"].astype(str).drop_duplicates().tolist()
    ordered_models = [model for model in model_order if model in present_models]
    ordered_models.extend(model for model in present_models if model not in ordered_models)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x_positions = {model_raw: idx for idx, model_raw in enumerate(ordered_models)}
    structure_colors = _build_structure_color_map(structure_order)

    if len(structure_order) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-structure_offset, structure_offset, num=len(structure_order))
    structure_offsets = {
        structure: offsets[idx]
        for idx, structure in enumerate(structure_order)
    }

    for structure in structure_order:
        structure_rows = structure_summary[
            structure_summary["AlternativeStructure"].astype(str) == structure
        ].copy()
        if structure_rows.empty:
            continue

        x_values = [
            float(x_positions[str(model_raw)]) + float(structure_offsets[structure])
            for model_raw in structure_rows["NextWordPredictionModelRaw"]
        ]
        ax.scatter(
            x_values,
            structure_rows[structure_value_col],
            s=point_size,
            color=structure_colors[structure],
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )

    plotted_model_summary = model_summary[
        model_summary["NextWordPredictionModelRaw"].isin(ordered_models)
    ].copy()
    plotted_model_summary["_model_order"] = plotted_model_summary["NextWordPredictionModelRaw"].map(x_positions)
    plotted_model_summary = plotted_model_summary.sort_values("_model_order", ignore_index=True)
    ax.scatter(
        [x_positions[str(model_raw)] for model_raw in plotted_model_summary["NextWordPredictionModelRaw"]],
        plotted_model_summary[model_value_col],
        s=mean_point_size,
        color="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    label_lookup = (
        plotted_model_summary[["NextWordPredictionModelRaw", "NextWordPredictionModel"]]
        .drop_duplicates()
        .set_index("NextWordPredictionModelRaw")["NextWordPredictionModel"]
        .to_dict()
    )
    ax.set_xlim(-0.5, len(ordered_models) - 0.5)
    ax.set_xticks(
        list(range(len(ordered_models))),
        [label_lookup.get(model_raw, _pretty_model_name(model_raw)) for model_raw in ordered_models],
    )
    ax.set_xlabel("Next Word Prediction Model")
    ax.set_ylabel(y_label)
    ax.set_title(title if title is not None else default_title)
    ax.legend(
        handles=_legend_handles_for_structures(structure_order, structure_colors),
        title="Alternative Structure",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    _style_axes(ax)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_results_dir = root_dir / "focus_alt_exp_pipeline" / "results"

    parser = argparse.ArgumentParser(
        description="Summarize average log likelihood by context and alternative structure"
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs (default: <results-dir>/plots)",
    )
    parser.add_argument(
        "--structures",
        type=str,
        default=",".join(DEFAULT_STRUCTURE_FILTER),
        help="Comma-separated structures to include",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_FILTER),
        help="Comma-separated next-word prediction models to include",
    )
    parser.add_argument(
        "--model-order",
        type=str,
        default=",".join(DEFAULT_MODEL_FILTER),
        help="Comma-separated model order using raw names such as cloze,frequency",
    )
    parser.add_argument(
        "--align-frequency-to-cloze",
        action="store_true",
        help="Restrict frequency rows to the context/query/trigger combinations observed in cloze.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the across-model average log likelihood plot.",
    )
    parser.add_argument(
        "--correlation-title",
        type=str,
        default=None,
        help="Optional custom title for the across-model correlation plot.",
    )
    parser.add_argument(
        "--human-data",
        type=Path,
        default=_resolve_human_data_default(),
        help="Human participant-level CSV used to compute human negation probabilities.",
    )
    parser.add_argument("--point-size", type=float, default=95.0)
    parser.add_argument("--mean-point-size", type=float, default=120.0)
    parser.add_argument(
        "--structure-offset",
        type=float,
        default=0.18,
        help="Horizontal offset used to separate alternative-structure dots within each model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    structures = _parse_csv_list(args.structures)
    if not structures:
        raise ValueError("--structures produced an empty list")

    selected_models = _parse_csv_list(args.models)
    model_order = _parse_csv_list(args.model_order)

    out_dir = args.output_dir if args.output_dir is not None else args.results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = _collect_results(
        args.results_dir,
        structures,
        model_names=selected_models if selected_models else None,
    )
    if args.align_frequency_to_cloze:
        all_results = _align_frequency_to_cloze(all_results)

    summary = _summarize_results_by_context(all_results)

    if model_order:
        order_lookup = {_pretty_model_name(raw): idx for idx, raw in enumerate(model_order)}
        summary["_model_order"] = summary["NextWordPredictionModel"].map(order_lookup).fillna(10**6)
    else:
        summary["_model_order"] = 10**6

    summary["AlternativeStructure"] = pd.Categorical(
        summary["AlternativeStructure"],
        categories=structures,
        ordered=True,
    )
    structure_mean_summary = _summarize_structure_means(
        summary,
        structure_order=structures,
    )
    model_mean_summary = _summarize_model_means(structure_mean_summary)
    correlation_summary = _build_correlation_summary(
        all_results,
        structure_order=structures,
        human_data_path=args.human_data,
    )
    structure_correlation_summary = _summarize_structure_correlations(
        correlation_summary,
        structure_order=structures,
    )
    model_correlation_summary = _summarize_model_correlation_means(
        structure_correlation_summary,
    )

    summary = summary.sort_values(
        ["_model_order", "NextWordPredictionModel", "AlternativeStructure", "context"],
        ignore_index=True,
    ).drop(columns="_model_order")
    structure_mean_summary["_model_order"] = (
        structure_mean_summary["NextWordPredictionModel"].map(order_lookup).fillna(10**6)
        if model_order
        else 10**6
    )
    structure_mean_summary = structure_mean_summary.sort_values(
        ["_model_order", "NextWordPredictionModel", "AlternativeStructure"],
        ignore_index=True,
    ).drop(columns="_model_order")
    model_mean_summary["_model_order"] = (
        model_mean_summary["NextWordPredictionModel"].map(order_lookup).fillna(10**6)
        if model_order
        else 10**6
    )
    model_mean_summary = model_mean_summary.sort_values(
        ["_model_order", "NextWordPredictionModel"],
        ignore_index=True,
    ).drop(columns="_model_order")
    structure_correlation_summary["_model_order"] = (
        structure_correlation_summary["NextWordPredictionModel"].map(order_lookup).fillna(10**6)
        if model_order
        else 10**6
    )
    structure_correlation_summary = structure_correlation_summary.sort_values(
        ["_model_order", "NextWordPredictionModel", "AlternativeStructure"],
        ignore_index=True,
    ).drop(columns="_model_order")
    model_correlation_summary["_model_order"] = (
        model_correlation_summary["NextWordPredictionModel"].map(order_lookup).fillna(10**6)
        if model_order
        else 10**6
    )
    model_correlation_summary = model_correlation_summary.sort_values(
        ["_model_order", "NextWordPredictionModel"],
        ignore_index=True,
    ).drop(columns="_model_order")

    combined_csv = out_dir / "average_log_likelihood_by_context_and_alternative_structure.csv"
    structure_mean_csv = out_dir / "mean_log_likelihood_by_model_and_structure.csv"
    model_mean_csv = out_dir / "mean_log_likelihood_by_model.csv"
    model_plot = out_dir / "log_likelihood_by_model_with_structure_dots.png"
    correlation_points_csv = out_dir / "negation_probability_correlation_points.csv"
    structure_correlation_csv = out_dir / "negation_probability_correlation_by_model_and_structure.csv"
    model_correlation_csv = out_dir / "negation_probability_correlation_by_model.csv"
    correlation_plot = out_dir / "negation_probability_correlation_by_model_with_structure_dots.png"
    summary.to_csv(combined_csv, index=False)
    structure_mean_summary.to_csv(structure_mean_csv, index=False)
    model_mean_summary.to_csv(model_mean_csv, index=False)
    correlation_summary.to_csv(correlation_points_csv, index=False)
    structure_correlation_summary.to_csv(structure_correlation_csv, index=False)
    model_correlation_summary.to_csv(model_correlation_csv, index=False)
    _make_across_model_plot(
        structure_mean_summary,
        model_mean_summary,
        model_order=model_order if model_order else selected_models,
        structure_order=structures,
        output_path=model_plot,
        title=args.title,
        point_size=args.point_size,
        mean_point_size=args.mean_point_size,
        structure_offset=args.structure_offset,
        structure_value_col="mean_log_likelihood",
        model_value_col="overall_mean_log_likelihood",
        y_label="Average Log Likelihood",
        default_title="Average Log Likelihood by Next-Word Model",
    )
    _make_across_model_plot(
        structure_correlation_summary,
        model_correlation_summary,
        model_order=model_order if model_order else selected_models,
        structure_order=structures,
        output_path=correlation_plot,
        title=args.correlation_title,
        point_size=args.point_size,
        mean_point_size=args.mean_point_size,
        structure_offset=args.structure_offset,
        structure_value_col="pearson_correlation",
        model_value_col="mean_pearson_correlation",
        y_label="Pearson Correlation (r)",
        default_title="Negation Probability Correlation by Next-Word Model",
    )
    print("Saved:", combined_csv)
    print("Saved:", structure_mean_csv)
    print("Saved:", model_mean_csv)
    print("Saved:", model_plot)
    print("Saved:", correlation_points_csv)
    print("Saved:", structure_correlation_csv)
    print("Saved:", model_correlation_csv)
    print("Saved:", correlation_plot)

    for model_raw in summary["NextWordPredictionModelRaw"].drop_duplicates():
        model_csv = out_dir / (
            f"average_log_likelihood_by_context_and_alternative_structure__{_safe_stem(model_raw)}.csv"
        )
        summary[summary["NextWordPredictionModelRaw"] == model_raw].to_csv(model_csv, index=False)
        print("Saved:", model_csv)


if __name__ == "__main__":
    main()
