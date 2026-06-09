"""Plot Qwen trigger/query analysis results for presentation."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "trigger_analysis" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "plots"
DEFAULT_HEATMAP_CONTEXTS = ["fridge", "cold", "restaurant", "throw"]

SCORE_SPECS = {
    "sum_logprob": {
        "scatter_x": "base_logprob_sum",
        "scatter_y": "mean_but_not_logprob_sum",
        "pair_delta": "delta_logprob_sum",
        "x_label": "Base prompt log P(query)",
        "y_label": "Mean trigger prompt log P(query)",
        "heatmap_label": "Delta log P(query)",
        "file_suffix": "sum_logprob",
    },
    "mean_logprob_per_token": {
        "scatter_x": "base_logprob_mean",
        "scatter_y": "mean_but_not_logprob_mean",
        "pair_delta": "delta_logprob_mean",
        "x_label": "Base prompt mean token log P(query)",
        "y_label": "Mean trigger prompt mean token log P(query)",
        "heatmap_label": "Delta mean token log P(query)",
        "file_suffix": "mean_token_logprob",
    },
}


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _configure_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "trigger_analysis_mplconfig"),
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        str(Path(tempfile.gettempdir()) / "trigger_analysis_cache"),
    )
    import matplotlib

    matplotlib.use("Agg")


def _format_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, path: Path) -> None:
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _load_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    df = pd.read_csv(path)
    _require_columns(df, required_columns, path=path)
    return df


def _safe_stem(text: str) -> str:
    keep = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _save_figure(fig, output_dir: Path, stem: str, formats: list[str], *, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[plot] wrote {path}")


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.22)


def _context_color_map(contexts: Iterable[str]) -> dict[str, object]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return {
        context: cmap(idx % cmap.N)
        for idx, context in enumerate(sorted(contexts))
    }


def _pooled_correlation_text(
    correlations: pd.DataFrame,
    *,
    analysis_level: str,
    score_scale: str,
) -> str:
    subset = correlations[
        (correlations["analysis_level"] == analysis_level)
        & (correlations["context"] == "ALL")
        & (correlations["score_scale"] == score_scale)
    ].copy()
    if subset.empty:
        return ""

    pieces = []
    for method, label in [("pearson", "Pearson r"), ("spearman", "Spearman rho")]:
        row = subset[subset["method"] == method]
        if row.empty:
            continue
        value = float(row["correlation"].iloc[0])
        n = int(row["n"].iloc[0])
        pieces.append(f"{label} = {_format_number(value)}")

    if not pieces:
        return ""
    pieces.append(f"n = {n}")
    return "\n".join(pieces)


def plot_unique_query_scatter(
    *,
    query_scores: pd.DataFrame,
    query_correlations: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    spec = SCORE_SPECS[score_scale]
    x_col = spec["scatter_x"]
    y_col = spec["scatter_y"]
    required = ["story", "query", x_col, y_col]
    _require_columns(query_scores, required, path=Path("qwen_unique_query_logprobs.csv"))

    plot_df = query_scores[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    contexts = sorted(plot_df["story"].unique())
    colors = _context_color_map(contexts)

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for context, subset in plot_df.groupby("story", sort=True):
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=54,
            alpha=0.86,
            color=colors[context],
            edgecolor="white",
            linewidth=0.6,
            label=context,
        )

    min_value = float(min(plot_df[x_col].min(), plot_df[y_col].min()))
    max_value = float(max(plot_df[x_col].max(), plot_df[y_col].max()))
    padding = (max_value - min_value) * 0.06 if max_value > min_value else 1.0
    lims = (min_value - padding, max_value + padding)
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.2, alpha=0.55)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    annotation = _pooled_correlation_text(
        query_correlations,
        analysis_level="unique_query",
        score_scale=score_scale,
    )
    if annotation:
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#c9c9c9",
                "alpha": 0.92,
            },
        )

    ax.set_title("Base vs. Trigger-Conditioned Qwen Query Scores", fontsize=14, pad=14)
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    _style_axes(ax)
    ax.legend(
        title="Context",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    _save_figure(
        fig,
        output_dir,
        f"unique_query_scatter_{spec['file_suffix']}",
        formats,
        dpi=dpi,
    )
    plt.close(fig)


def plot_context_correlation_dotplot(
    *,
    query_correlations: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    required = [
        "analysis_level",
        "context",
        "score_scale",
        "method",
        "n",
        "correlation",
    ]
    _require_columns(query_correlations, required, path=Path("qwen_unique_query_correlations.csv"))
    plot_df = query_correlations[
        (query_correlations["analysis_level"] == "unique_query")
        & (query_correlations["score_scale"] == score_scale)
        & (query_correlations["context"] != "ALL")
        & (query_correlations["method"].isin(["pearson", "spearman"]))
    ].copy()
    if plot_df.empty:
        raise ValueError("No per-context unique-query correlation rows to plot.")

    pivot = plot_df.pivot(index="context", columns="method", values="correlation")
    sort_col = "pearson" if "pearson" in pivot.columns else pivot.columns[0]
    contexts = pivot.sort_values(sort_col, ascending=True).index.tolist()
    y_positions = np.arange(len(contexts))

    fig_height = max(5.2, 0.36 * len(contexts) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    ax.axvline(0, color="#6b6b6b", linewidth=1.0)

    method_specs = {
        "pearson": {"label": "Pearson", "offset": -0.12, "color": "#3268a8", "marker": "o"},
        "spearman": {"label": "Spearman", "offset": 0.12, "color": "#c2553d", "marker": "s"},
    }
    for method, method_spec in method_specs.items():
        if method not in pivot.columns:
            continue
        values = pivot.loc[contexts, method].astype(float).to_numpy()
        ax.scatter(
            values,
            y_positions + method_spec["offset"],
            s=58,
            color=method_spec["color"],
            marker=method_spec["marker"],
            label=method_spec["label"],
            zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(contexts)
    ax.set_xlim(-1.0, 1.1)
    ax.set_xlabel("Correlation between base and mean trigger-conditioned query scores")
    ax.set_title("Unique-Query Correlations by Context", fontsize=14, pad=12)
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.12)

    _save_figure(
        fig,
        output_dir,
        f"unique_query_context_correlations_{SCORE_SPECS[score_scale]['file_suffix']}",
        formats,
        dpi=dpi,
    )
    plt.close(fig)


def _ordered_context_alternatives(context_rows: pd.DataFrame, delta_col: str) -> list[str]:
    query_scores = (
        context_rows[["query", "base_logprob_sum", "base_logprob_mean"]]
        .drop_duplicates("query")
        .copy()
    )
    base_col = "base_logprob_mean" if delta_col.endswith("_mean") else "base_logprob_sum"
    query_scores = query_scores.sort_values(base_col, ascending=False)
    ordered_queries = query_scores["query"].astype(str).tolist()

    triggers = context_rows["trigger"].astype(str).unique().tolist()
    ordered = [value for value in ordered_queries if value in triggers]
    ordered.extend(sorted(set(triggers).difference(ordered)))
    return ordered


def plot_delta_heatmaps(
    *,
    pair_scores: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    heatmap_contexts: list[str],
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    delta_col = SCORE_SPECS[score_scale]["pair_delta"]
    required = ["story", "trigger", "query", delta_col, "base_logprob_sum", "base_logprob_mean"]
    _require_columns(pair_scores, required, path=Path("qwen_trigger_query_pair_logprobs.csv"))

    selected_contexts = [
        context
        for context in heatmap_contexts
        if context in set(pair_scores["story"].astype(str).unique())
    ]
    missing_contexts = sorted(set(heatmap_contexts).difference(selected_contexts))
    if missing_contexts:
        print(f"[warn] missing heatmap contexts skipped: {','.join(missing_contexts)}")
    if not selected_contexts:
        raise ValueError("No requested heatmap contexts are available in pair scores.")

    selected = pair_scores[pair_scores["story"].isin(selected_contexts)].copy()
    max_abs = float(np.nanmax(np.abs(selected[delta_col].astype(float).to_numpy())))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    cols = 2 if len(selected_contexts) > 1 else 1
    rows = math.ceil(len(selected_contexts) / cols)
    fig_width = 6.0 * cols
    fig_height = 4.8 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    cmap = plt.get_cmap("vlag").copy()
    cmap.set_bad("#f2f2f2")

    for idx, (ax, context) in enumerate(zip(axes.ravel(), selected_contexts)):
        context_rows = pair_scores[pair_scores["story"] == context].copy()
        order = _ordered_context_alternatives(context_rows, delta_col)
        heatmap_data = context_rows.pivot(index="trigger", columns="query", values=delta_col)
        heatmap_data = heatmap_data.reindex(index=order, columns=order)

        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.7,
            linecolor="white",
            annot=True,
            fmt=".1f",
            cbar=False,
            square=True,
            mask=heatmap_data.isna(),
        )
        ax.set_title(context, fontsize=13, pad=10)
        row_idx = idx // cols
        ax.set_xlabel("Query" if row_idx == rows - 1 else "")
        ax.set_ylabel("Trigger")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    for ax in axes.ravel()[len(selected_contexts) :]:
        ax.axis("off")

    norm = plt.Normalize(vmin=-max_abs, vmax=max_abs)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(SCORE_SPECS[score_scale]["heatmap_label"])

    fig.suptitle("Effect of Trigger Prompt on Query Log Probability", fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.01,
        "Positive values mean the query is more likely after '{trigger} but not'; blank cells are trigger=query.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.13,
        left=0.08,
        right=0.88,
        hspace=0.55,
        wspace=0.32,
    )

    _save_figure(
        fig,
        output_dir,
        f"delta_heatmaps_{SCORE_SPECS[score_scale]['file_suffix']}",
        formats,
        dpi=dpi,
    )
    plt.close(fig)


def plot_individual_delta_heatmaps(
    *,
    pair_scores: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    delta_col = SCORE_SPECS[score_scale]["pair_delta"]
    required = ["story", "trigger", "query", delta_col, "base_logprob_sum", "base_logprob_mean"]
    _require_columns(pair_scores, required, path=Path("qwen_trigger_query_pair_logprobs.csv"))

    max_abs = float(np.nanmax(np.abs(pair_scores[delta_col].astype(float).to_numpy())))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    cmap = plt.get_cmap("vlag").copy()
    cmap.set_bad("#f2f2f2")
    context_dir = output_dir / "context_heatmaps"

    for context in sorted(pair_scores["story"].astype(str).unique()):
        context_rows = pair_scores[pair_scores["story"] == context].copy()
        order = _ordered_context_alternatives(context_rows, delta_col)
        heatmap_data = context_rows.pivot(index="trigger", columns="query", values=delta_col)
        heatmap_data = heatmap_data.reindex(index=order, columns=order)

        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            linewidths=0.7,
            linecolor="white",
            annot=True,
            fmt=".1f",
            cbar=True,
            square=True,
            mask=heatmap_data.isna(),
            cbar_kws={"label": SCORE_SPECS[score_scale]["heatmap_label"]},
        )
        ax.set_title(f"{context}: trigger effect on query log probability", fontsize=13, pad=10)
        ax.set_xlabel("Query", labelpad=24)
        ax.set_ylabel("Trigger")
        ax.tick_params(axis="x", rotation=35, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
        fig.text(
            0.5,
            0.035,
            "Positive values mean the query is more likely after '{trigger} but not'; blank cells are trigger=query.",
            ha="center",
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.35, left=0.18, right=0.96, top=0.87)

        _save_figure(
            fig,
            context_dir,
            f"delta_heatmap_{_safe_stem(context)}_{SCORE_SPECS[score_scale]['file_suffix']}",
            formats,
            dpi=dpi,
        )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Qwen trigger/query results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--score-scale",
        choices=sorted(SCORE_SPECS),
        default="sum_logprob",
        help="Use summed continuation logprob by default; mean token logprob is a sensitivity view.",
    )
    parser.add_argument(
        "--heatmap-contexts",
        type=str,
        default=",".join(DEFAULT_HEATMAP_CONTEXTS),
        help="Comma-separated contexts to include in the heatmap panel.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated output formats passed to matplotlib savefig.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    formats = _parse_csv_list(args.formats)
    if not formats:
        raise ValueError("--formats must include at least one format.")

    heatmap_contexts = _parse_csv_list(args.heatmap_contexts)
    if not heatmap_contexts:
        raise ValueError("--heatmap-contexts must include at least one context.")

    pair_scores = _load_csv(
        args.results_dir / "qwen_trigger_query_pair_logprobs.csv",
        ["story", "trigger", "query"],
    )
    query_scores = _load_csv(
        args.results_dir / "qwen_unique_query_logprobs.csv",
        ["story", "query"],
    )
    query_correlations = _load_csv(
        args.results_dir / "qwen_unique_query_correlations.csv",
        ["analysis_level", "context", "score_scale", "method", "correlation"],
    )

    plot_unique_query_scatter(
        query_scores=query_scores,
        query_correlations=query_correlations,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        formats=formats,
        dpi=args.dpi,
    )
    plot_context_correlation_dotplot(
        query_correlations=query_correlations,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        formats=formats,
        dpi=args.dpi,
    )
    plot_delta_heatmaps(
        pair_scores=pair_scores,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        heatmap_contexts=heatmap_contexts,
        formats=formats,
        dpi=args.dpi,
    )
    plot_individual_delta_heatmaps(
        pair_scores=pair_scores,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        formats=formats,
        dpi=args.dpi,
    )

    print("[complete] trigger analysis plots generated")


if __name__ == "__main__":
    main()
