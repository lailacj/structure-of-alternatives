"""Generate per-context negation-probability heatmaps from result CSVs.

For each next-word prediction model (for example, ``cloze`` or ``frequency``),
this script creates one figure per context. Each figure contains a grid of
heatmaps for the alternative structures (set, ordering, disjunction,
conjunction) plus one additional heatmap for the human responses. Within each
heatmap:

- the x axis is the query word
- the y axis is the trigger word
- the cell value is the aggregated negation probability

Because the result CSVs can contain repeated experimental rows for the same
``context``/``trigger``/``query``/``set_boundary`` combination, the script first
averages duplicate rows within each boundary and then aggregates across
boundaries to produce one value per heatmap cell. The human-response panel uses
the same aggregation strategy on the ``neg`` column.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILE_PATTERN = re.compile(r"^(set|ordering|conjunction|disjunction)_results_(.+)\.csv$")
DEFAULT_STRUCTURE_ORDER = ["set", "ordering", "disjunction", "conjunction"]
REQUIRED_COLUMNS = {
    "set_boundary",
    "context",
    "trigger",
    "query",
    "neg",
    "negation_probability",
}


def _parse_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _safe_stem(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text))
    return safe.strip("_")


def _pretty_model_name(raw_name: str) -> str:
    acronym_map = {
        "frequency": "Frequency",
        "qwen": "Qwen",
        "cloze": "Cloze",
    }
    parts = raw_name.replace("-", "_").split("_")
    return " ".join(acronym_map.get(part.lower(), part.capitalize()) for part in parts)


def _pretty_structure_name(name: str) -> str:
    return {
        "set": "Set",
        "ordering": "Ordering",
        "disjunction": "Disjunction",
        "conjunction": "Conjunction",
    }.get(name, name.title())


def _build_parser() -> argparse.ArgumentParser:
    default_results_dir = (
        Path(__file__).resolve().parent.parent / "results"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Create per-context negation-probability heatmaps from "
            "focus_alt_exp result CSVs."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help="Directory containing *_results_<model>.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for output figures. Defaults to "
            "<results-dir>/plots/negation_probability_heatmaps."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model suffixes to include "
            "(for example: cloze,frequency). Defaults to all models found."
        ),
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default=None,
        help=(
            "Comma-separated list of contexts to include. "
            "Defaults to all contexts found for each model."
        ),
    )
    parser.add_argument(
        "--structures",
        type=str,
        default=",".join(DEFAULT_STRUCTURE_ORDER),
        help=(
            "Comma-separated list of structures to include. "
            f"Defaults to {','.join(DEFAULT_STRUCTURE_ORDER)}."
        ),
    )
    parser.add_argument(
        "--boundary-agg",
        choices=["mean", "median", "min", "max"],
        default="mean",
        help=(
            "How to aggregate negation_probability across distinct "
            "set_boundary values after duplicate rows are collapsed."
        ),
    )
    parser.add_argument(
        "--word-order",
        choices=["appearance", "alphabetical"],
        default="appearance",
        help="How to order words on the axes within each context.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure resolution in dots per inch.",
    )
    return parser


def _collect_results(
    results_dir: Path,
    structures: Iterable[str],
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    structure_order = {name: index for index, name in enumerate(structures)}
    requested_models = set(model_names) if model_names else None
    frames = []

    for path in sorted(results_dir.rglob("*_results_*.csv")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        structure, model_raw = match.groups()
        if structure not in structure_order:
            continue
        if requested_models is not None and model_raw not in requested_models:
            continue

        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {path}: {missing_str}")

        frame = df.copy()
        frame["AlternativeStructure"] = structure
        frame["NextWordPredictionModelRaw"] = model_raw
        frame["NextWordPredictionModel"] = _pretty_model_name(model_raw)
        frame["_structure_order"] = structure_order[structure]
        frame["_source_row"] = np.arange(len(frame))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No matching result CSVs found in {results_dir} for structures "
            f"{list(structures)}"
        )

    return pd.concat(frames, ignore_index=True)


def _aggregate_cells(results_df: pd.DataFrame, boundary_agg: str) -> pd.DataFrame:
    boundary_level = (
        results_df.groupby(
            [
                "NextWordPredictionModelRaw",
                "NextWordPredictionModel",
                "AlternativeStructure",
                "context",
                "set_boundary",
                "trigger",
                "query",
            ],
            as_index=False,
            sort=False,
        )["negation_probability"]
        .mean()
    )

    cell_level = (
        boundary_level.groupby(
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
        )
        .agg(
            negation_probability=("negation_probability", boundary_agg),
            boundary_count=("set_boundary", "nunique"),
        )
    )
    return cell_level


def _get_word_order(context_df: pd.DataFrame, mode: str) -> List[str]:
    words = pd.unique(
        pd.concat([context_df["trigger"], context_df["query"]], ignore_index=True)
    )
    labels = [str(word) for word in words if pd.notna(word)]
    if mode == "alphabetical":
        return sorted(labels, key=lambda item: item.lower())
    return labels


def _build_heatmap_matrix(
    context_cells: pd.DataFrame,
    words: List[str],
    value_column: str = "negation_probability",
) -> pd.DataFrame:
    matrix = context_cells.pivot(
        index="trigger",
        columns="query",
        values=value_column,
    )
    return matrix.reindex(index=words, columns=words)


def _style_heatmap_axes(ax: plt.Axes, words: List[str]) -> None:
    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Query word")
    ax.set_ylabel("Trigger word")
    ax.set_xticks(np.arange(-0.5, len(words), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(words), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)


def _annotate_heatmap(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    values = matrix.to_numpy(dtype=float)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                continue
            text_color = "white" if value >= 0.5 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )


def _aggregate_human_cells(context_df: pd.DataFrame) -> pd.DataFrame:
    preferred_structure = "ordering"
    available_structures = set(context_df["AlternativeStructure"].unique())
    if preferred_structure in available_structures:
        human_source = context_df[context_df["AlternativeStructure"] == preferred_structure]
    else:
        first_structure = str(context_df["AlternativeStructure"].iloc[0])
        human_source = context_df[context_df["AlternativeStructure"] == first_structure]

    boundary_level = (
        human_source.groupby(
            ["context", "set_boundary", "trigger", "query"],
            as_index=False,
            sort=False,
        )["neg"]
        .mean()
    )

    return (
        boundary_level.groupby(
            ["context", "trigger", "query"],
            as_index=False,
            sort=False,
        )["neg"]
        .mean()
        .rename(columns={"neg": "human_response"})
    )


def _plot_context_figure(
    *,
    raw_context_df: pd.DataFrame,
    aggregated_context_df: pd.DataFrame,
    model_raw: str,
    context: str,
    structures: List[str],
    boundary_agg: str,
    word_order: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    words = _get_word_order(raw_context_df, mode=word_order)
    human_context_df = _aggregate_human_cells(raw_context_df)
    plot_specs = [
        (structure, _pretty_structure_name(structure), "negation_probability")
        for structure in structures
    ]
    plot_specs.append(("human", "Human Responses", "human_response"))

    n_plots = len(plot_specs)
    ncols = 2
    nrows = 3
    subplot_size = max(3.8, 0.7 * len(words) + 2.8)
    figsize = (ncols * subplot_size + 2.0, nrows * subplot_size + 1.8)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )

    cmap = plt.get_cmap("autumn_r").copy()
    cmap.set_bad("#ececec")
    flat_axes = list(axes.flat)
    image = None
    plotted_axes: List[plt.Axes] = []

    for index, (panel_key, panel_title, value_column) in enumerate(plot_specs):
        ax = flat_axes[index]
        if panel_key == "human":
            panel_df = human_context_df
        else:
            panel_df = aggregated_context_df[
                aggregated_context_df["AlternativeStructure"] == panel_key
            ]
        matrix = _build_heatmap_matrix(panel_df, words, value_column=value_column)
        masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
        image = ax.imshow(
            masked,
            vmin=0.0,
            vmax=1.0,
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_facecolor("#ececec")
        ax.set_title(panel_title)
        _style_heatmap_axes(ax, words)
        _annotate_heatmap(ax, matrix)
        plotted_axes.append(ax)

    for ax in flat_axes[n_plots:]:
        ax.set_visible(False)

    if image is None:
        raise ValueError(f"No heatmaps were drawn for context '{context}' and model '{model_raw}'")

    colorbar = fig.colorbar(image, ax=plotted_axes, shrink=0.9)
    colorbar.set_label("Negation probability")

    fig.suptitle(
        (
            f"{_pretty_model_name(model_raw)}: {context}\n"
            f"Cells aggregate duplicate rows within a boundary, then use "
            f"{boundary_agg} across set boundaries"
        ),
        fontsize=14,
    )

    model_output_dir = output_dir / _safe_stem(model_raw)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / f"{_safe_stem(context)}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (results_dir / "plots" / "negation_probability_heatmaps").resolve()
    )
    structures = _parse_csv_list(args.structures)
    if not structures:
        raise ValueError("At least one structure must be provided via --structures")

    requested_models = _parse_csv_list(args.models)
    requested_contexts = set(_parse_csv_list(args.contexts))

    raw_results = _collect_results(
        results_dir=results_dir,
        structures=structures,
        model_names=requested_models or None,
    )
    aggregated = _aggregate_cells(raw_results, boundary_agg=args.boundary_agg)

    model_order = (
        requested_models
        if requested_models
        else sorted(raw_results["NextWordPredictionModelRaw"].unique())
    )

    figures_written = 0
    for model_raw in model_order:
        model_raw_results = raw_results[
            raw_results["NextWordPredictionModelRaw"] == model_raw
        ].sort_values(
            by=["_structure_order", "_source_row"],
            kind="stable",
        )
        if model_raw_results.empty:
            continue

        available_contexts = list(pd.unique(model_raw_results["context"]))
        contexts = [
            context
            for context in available_contexts
            if not requested_contexts or context in requested_contexts
        ]

        for context in contexts:
            raw_context_df = model_raw_results[model_raw_results["context"] == context]
            aggregated_context_df = aggregated[
                (aggregated["NextWordPredictionModelRaw"] == model_raw)
                & (aggregated["context"] == context)
            ]
            if aggregated_context_df.empty:
                continue

            output_path = _plot_context_figure(
                raw_context_df=raw_context_df,
                aggregated_context_df=aggregated_context_df,
                model_raw=model_raw,
                context=context,
                structures=structures,
                boundary_agg=args.boundary_agg,
                word_order=args.word_order,
                output_dir=output_dir,
                dpi=args.dpi,
            )
            figures_written += 1
            print(f"Wrote {output_path}")

    if figures_written == 0:
        raise FileNotFoundError(
            "No figures were written. Check the requested models, contexts, and structures."
        )

    print(f"Wrote {figures_written} figure(s) to {output_dir}")


if __name__ == "__main__":
    main()
