"""Plot Qwen trigger/query log-probability differences.

This script uses the logprob CSVs produced by
score_qwen_trigger_query_correlations.py and visualizes:

    trigger-conditioned log P(query) - base log P(query)

at both the unique-query level and the trigger/query-pair level.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "trigger_analysis" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "plots" / "logprob_differences"

SCORE_SPECS = {
    "sum_logprob": {
        "base_col": "base_logprob_sum",
        "query_delta_col": "mean_delta_logprob_sum",
        "pair_delta_col": "delta_logprob_sum",
        "x_label": "Base prompt log P(query)",
        "query_delta_label": "Mean trigger effect on log P(query)",
        "pair_delta_label": "Trigger effect on log P(query)",
        "summary_label": "summed logprob",
        "file_suffix": "sum_logprob",
    },
    "mean_logprob_per_token": {
        "base_col": "base_logprob_mean",
        "query_delta_col": "mean_delta_logprob_mean",
        "pair_delta_col": "delta_logprob_mean",
        "x_label": "Base prompt mean token log P(query)",
        "query_delta_label": "Mean trigger effect on mean token log P(query)",
        "pair_delta_label": "Trigger effect on mean token log P(query)",
        "summary_label": "mean token logprob",
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


def _format_number(value: float, digits: int = 2) -> str:
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
    return {context: cmap(idx % cmap.N) for idx, context in enumerate(sorted(contexts))}


def _finite_values(frame: pd.DataFrame, value_col: str) -> pd.Series:
    return (
        frame[value_col]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _summary_text(values: pd.Series) -> str:
    finite = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return "No finite values"

    positive = int((finite > 0).sum())
    total = int(len(finite))
    return "\n".join(
        [
            f"Mean delta = {_format_number(float(finite.mean()))}",
            f"Median delta = {_format_number(float(finite.median()))}",
            f"Positive = {positive}/{total} ({100 * positive / total:.1f}%)",
        ]
    )


def _context_summary(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    records = []
    for context, subset in frame.groupby("story", sort=False):
        values = _finite_values(subset, value_col)
        n = int(len(values))
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        sem = sd / np.sqrt(n) if n > 1 else float("nan")
        margin = 1.96 * sem if np.isfinite(sem) else float("nan")
        mean = float(values.mean()) if n else float("nan")
        records.append(
            {
                "story": str(context),
                "n": n,
                "mean": mean,
                "median": float(values.median()) if n else float("nan"),
                "sd": sd,
                "ci_low": mean - margin if np.isfinite(margin) else float("nan"),
                "ci_high": mean + margin if np.isfinite(margin) else float("nan"),
                "positive_fraction": float((values > 0).mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("mean", ascending=True)


def plot_unique_query_delta_scatter(
    *,
    query_scores: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    spec = SCORE_SPECS[score_scale]
    base_col = spec["base_col"]
    delta_col = spec["query_delta_col"]
    required = ["story", "query", base_col, delta_col]
    _require_columns(query_scores, required, path=Path("qwen_unique_query_logprobs.csv"))

    plot_df = (
        query_scores[required]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[base_col, delta_col])
        .copy()
    )
    contexts = sorted(plot_df["story"].astype(str).unique())
    colors = _context_color_map(contexts)

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for context, subset in plot_df.groupby("story", sort=True):
        ax.scatter(
            subset[base_col].astype(float),
            subset[delta_col].astype(float),
            s=56,
            alpha=0.86,
            color=colors[str(context)],
            edgecolor="white",
            linewidth=0.6,
            label=str(context),
    )

    ax.axhline(0, color="#333333", linestyle="--", linewidth=1.1, alpha=0.75)
    blended_transform = ax.get_yaxis_transform()
    ax.text(
        0.01,
        0,
        "no change",
        transform=blended_transform,
        fontsize=9,
        color="#555555",
        va="bottom",
    )

    annotation = _summary_text(plot_df[delta_col])
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

    ax.set_title("Qwen Trigger Effect by Unique Query", fontsize=14, pad=14)
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["query_delta_label"])
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
        f"unique_query_delta_scatter_{spec['file_suffix']}",
        formats,
        dpi=dpi,
    )
    plt.close(fig)


def plot_context_delta_dotplot(
    *,
    scores: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    value_col: str,
    y_label: str,
    file_stem: str,
    title: str,
    formats: list[str],
    dpi: int,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    required = ["story", value_col]
    _require_columns(scores, required, path=Path("logprob scores"))

    plot_df = scores[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    summary = _context_summary(plot_df, value_col)
    contexts = summary["story"].tolist()
    y_positions = np.arange(len(contexts))

    fig_height = max(5.2, 0.36 * len(contexts) + 1.4)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.1, alpha=0.75)

    for y_pos, context in zip(y_positions, contexts):
        values = _finite_values(plot_df[plot_df["story"] == context], value_col)
        if values.empty:
            continue
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0])
        ax.scatter(
            values.to_numpy(),
            np.full(len(values), y_pos) + offsets,
            s=22,
            color="#8d99a6",
            alpha=0.45,
            linewidth=0,
            zorder=2,
        )

    xerr_low = summary["mean"] - summary["ci_low"]
    xerr_high = summary["ci_high"] - summary["mean"]
    xerr = np.vstack(
        [
            xerr_low.fillna(0).to_numpy(dtype=float),
            xerr_high.fillna(0).to_numpy(dtype=float),
        ]
    )
    colors = np.where(summary["mean"] >= 0, "#2f7d5c", "#b35a4b")
    ax.errorbar(
        summary["mean"],
        y_positions,
        xerr=xerr,
        fmt="none",
        ecolor="#323232",
        elinewidth=1.2,
        capsize=3,
        alpha=0.88,
        zorder=3,
    )
    ax.scatter(
        summary["mean"],
        y_positions,
        s=72,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Context mean",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(contexts)
    ax.set_xlabel(y_label)
    ax.set_title(title, fontsize=14, pad=12)
    _style_axes(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.12)

    _save_figure(
        fig,
        output_dir,
        f"{file_stem}_{SCORE_SPECS[score_scale]['file_suffix']}",
        formats,
        dpi=dpi,
    )
    plt.close(fig)
    return summary


def plot_trigger_delta_distribution(
    *,
    pair_scores: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
    formats: list[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    spec = SCORE_SPECS[score_scale]
    delta_col = spec["pair_delta_col"]
    required = ["story", "trigger", "query", delta_col]
    _require_columns(pair_scores, required, path=Path("qwen_trigger_query_pair_logprobs.csv"))

    plot_df = (
        pair_scores[required]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[delta_col])
        .copy()
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.hist(
        plot_df[delta_col].astype(float),
        bins=28,
        color="#5d7f9f",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.axvline(
        float(plot_df[delta_col].mean()),
        color="#b35a4b",
        linewidth=1.6,
    )
    ax.text(
        0.98,
        0.96,
        _summary_text(plot_df[delta_col]),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#c9c9c9",
            "alpha": 0.92,
        },
    )

    ax.set_title("Distribution of Trigger/Query Logprob Differences", fontsize=14, pad=12)
    ax.set_xlabel(spec["pair_delta_label"])
    ax.set_ylabel("Number of trigger/query pairs")
    _style_axes(ax)

    _save_figure(
        fig,
        output_dir,
        f"pair_delta_distribution_{spec['file_suffix']}",
        formats,
        dpi=dpi,
    )
    plt.close(fig)


def write_context_summary(
    *,
    query_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    output_dir: Path,
    score_scale: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = SCORE_SPECS[score_scale]

    merged = query_summary.add_prefix("unique_query_").rename(
        columns={"unique_query_story": "story"}
    )
    pair_prefixed = pair_summary.add_prefix("pair_").rename(columns={"pair_story": "story"})
    merged = merged.merge(pair_prefixed, on="story", how="outer")

    path = output_dir / f"context_delta_summary_{spec['file_suffix']}.csv"
    merged.to_csv(path, index=False)
    print(f"[summary] wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Qwen trigger/query logprob differences."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--score-scale",
        choices=sorted(SCORE_SPECS),
        default="sum_logprob",
        help="Use summed continuation logprob by default; mean token logprob is a sensitivity view.",
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

    spec = SCORE_SPECS[args.score_scale]
    query_scores = _load_csv(
        args.results_dir / "qwen_unique_query_logprobs.csv",
        ["story", "query", spec["base_col"], spec["query_delta_col"]],
    )
    pair_scores = _load_csv(
        args.results_dir / "qwen_trigger_query_pair_logprobs.csv",
        ["story", "trigger", "query", spec["pair_delta_col"]],
    )

    plot_unique_query_delta_scatter(
        query_scores=query_scores,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        formats=formats,
        dpi=args.dpi,
    )
    query_summary = plot_context_delta_dotplot(
        scores=query_scores,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        value_col=spec["query_delta_col"],
        y_label=spec["query_delta_label"],
        file_stem="unique_query_context_mean_delta",
        title="Mean Trigger Effect by Context and Query",
        formats=formats,
        dpi=args.dpi,
    )
    pair_summary = plot_context_delta_dotplot(
        scores=pair_scores,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        value_col=spec["pair_delta_col"],
        y_label=spec["pair_delta_label"],
        file_stem="pair_context_mean_delta",
        title="Trigger/Query Pair Effects by Context",
        formats=formats,
        dpi=args.dpi,
    )
    plot_trigger_delta_distribution(
        pair_scores=pair_scores,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
        formats=formats,
        dpi=args.dpi,
    )
    write_context_summary(
        query_summary=query_summary,
        pair_summary=pair_summary,
        output_dir=args.output_dir,
        score_scale=args.score_scale,
    )

    print(
        "[complete] logprob-difference plots generated "
        f"for {spec['summary_label']}"
    )


if __name__ == "__main__":
    main()
