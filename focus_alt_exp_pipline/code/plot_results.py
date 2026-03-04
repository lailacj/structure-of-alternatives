"""Plot summaries and diagnostics for focus_alt_exp result CSVs.

Generates:
1) Average log likelihood by NextWordPredictionModel with confidence intervals.
2) For each NextWordPredictionModel, average log likelihood by AlternativeStructure
   with confidence intervals.
3) Diagnostics CSVs covering missingness, dispersion, floor effects, and boundary
   imbalance.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILE_PATTERN = re.compile(r"^(set|ordering|conjunction|disjunction)_results_(.+)\.csv$")
DEFAULT_STRUCTURE_ORDER = ["set", "ordering", "conjunction", "disjunction"]
LOG_LIKELIHOOD_FLOOR = float(np.log(1e-10))
MISSING_SUMMARY_COLUMNS = [
    "missing_rows",
    "missing_query_rows",
    "missing_trigger_rows",
    "missing_boundaries",
    "missing_contexts",
]


def _parse_csv_list(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _pretty_model_name(raw_name: str) -> str:
    acronym_map = {
        "bert": "BERT",
        "bert_static": "Static BERT",
        "qwen": "Qwen",
        "cloze": "Cloze",
    }
    parts = raw_name.replace("-", "_").split("_")
    return " ".join(acronym_map.get(part.lower(), part.capitalize()) for part in parts)


def _safe_stem(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    return safe.strip("_")


def _collect_results(results_dir: Path, structures: Iterable[str]) -> pd.DataFrame:
    structure_set = set(structures)
    frames = []

    for path in sorted(results_dir.glob("*_results_*.csv")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        structure, model_raw = match.groups()
        if structure not in structure_set:
            continue

        df = pd.read_csv(path)
        if "log_likelihood" not in df.columns:
            raise ValueError(f"Missing 'log_likelihood' in {path}")

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


def _collect_missing_trials(results_dir: Path, model_names: Iterable[str]) -> pd.DataFrame:
    frames = []
    for model_raw in sorted(set(model_names)):
        path = results_dir / f"missing_trials_{model_raw}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        frame = df.copy()
        frame["NextWordPredictionModelRaw"] = model_raw
        frame["NextWordPredictionModel"] = _pretty_model_name(model_raw)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "set_boundary",
                "context",
                "trigger",
                "query",
                "reason",
                "NextWordPredictionModelRaw",
                "NextWordPredictionModel",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def _apply_model_order(
    summary_df: pd.DataFrame,
    model_order: List[str],
    extra_sort_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    sort_cols = ["NextWordPredictionModel"]
    with_order = summary_df.copy()

    if model_order:
        normalized_order = [item.strip() for item in model_order if item.strip()]
        order_lookup = {}
        for idx, raw in enumerate(normalized_order):
            pretty = _pretty_model_name(raw)
            order_lookup[pretty] = idx
        with_order["_order"] = with_order["NextWordPredictionModel"].map(order_lookup).fillna(10**6)
        sort_cols = ["_order", "NextWordPredictionModel"]

    if extra_sort_cols:
        sort_cols.extend(extra_sort_cols)

    with_order = with_order.sort_values(sort_cols, ignore_index=True)
    if "_order" in with_order.columns:
        with_order = with_order.drop(columns=["_order"])
    return with_order


def _style_axes() -> None:
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _bootstrap_mean_ci(
    group_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    mean_value = float(group_df["log_likelihood"].mean())
    if n_bootstrap <= 0:
        return mean_value, mean_value

    if "context" not in group_df.columns or group_df["context"].nunique() <= 1:
        return mean_value, mean_value

    context_stats = group_df.groupby("context", sort=False)["log_likelihood"].agg(["sum", "count"])
    context_sums = context_stats["sum"].to_numpy(dtype=float)
    context_counts = context_stats["count"].to_numpy(dtype=float)
    num_contexts = len(context_stats)

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, num_contexts, size=num_contexts)
        bootstrap_means[idx] = context_sums[sampled].sum() / context_counts[sampled].sum()

    alpha = (1.0 - ci_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _summarize_missing_trials(missing_trials: pd.DataFrame) -> pd.DataFrame:
    if missing_trials.empty:
        return pd.DataFrame(
            columns=[
                "NextWordPredictionModelRaw",
                "NextWordPredictionModel",
                *MISSING_SUMMARY_COLUMNS,
            ]
        )

    summary = (
        missing_trials.groupby(
            ["NextWordPredictionModelRaw", "NextWordPredictionModel"],
            as_index=False,
        )
        .agg(
            missing_rows=("reason", "size"),
            missing_boundaries=("set_boundary", "nunique"),
            missing_contexts=("context", "nunique"),
        )
    )

    reason_counts = (
        missing_trials.assign(
            is_query_missing=(missing_trials["reason"] == "query_not_in_context").astype(int),
            is_trigger_missing=(missing_trials["reason"] == "trigger_not_in_context").astype(int),
        )
        .groupby(["NextWordPredictionModelRaw", "NextWordPredictionModel"], as_index=False)
        .agg(
            missing_query_rows=("is_query_missing", "sum"),
            missing_trigger_rows=("is_trigger_missing", "sum"),
        )
    )

    return summary.merge(
        reason_counts,
        on=["NextWordPredictionModelRaw", "NextWordPredictionModel"],
        how="left",
    )


def _summarize_results(
    all_results: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    missing_summary: pd.DataFrame,
    n_bootstrap: int,
    ci_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows = []
    grouped = all_results.groupby(list(group_cols), sort=False, dropna=False)

    for idx, (group_key, group_df) in enumerate(grouped):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(zip(group_cols, group_key))
        values = group_df["log_likelihood"].to_numpy(dtype=float)
        floor_rows = int(np.isclose(values, LOG_LIKELIHOOD_FLOOR).sum())
        ci_lower, ci_upper = _bootstrap_mean_ci(
            group_df,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=bootstrap_seed + idx,
        )

        row["observed_rows"] = len(group_df)
        row["observed_boundaries"] = int(group_df["set_boundary"].nunique()) if "set_boundary" in group_df.columns else 0
        row["observed_contexts"] = int(group_df["context"].nunique()) if "context" in group_df.columns else 0
        row["average_log_likelihood"] = float(values.mean())
        row["median_log_likelihood"] = float(np.median(values))
        row["std_log_likelihood"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["min_log_likelihood"] = float(values.min())
        row["max_log_likelihood"] = float(values.max())
        row["log_floor_rows"] = floor_rows
        row["log_floor_rate"] = float(floor_rows / len(group_df))
        row["ci_lower"] = ci_lower
        row["ci_upper"] = ci_upper
        row["ci_level"] = ci_level
        row["bootstrap_samples"] = n_bootstrap

        if "probability_query_observed" in group_df.columns:
            zero_prob = int((group_df["probability_query_observed"].to_numpy(dtype=float) <= 1e-10).sum())
            row["zero_probability_rows"] = zero_prob
            row["zero_probability_rate"] = float(zero_prob / len(group_df))

        if "set_boundary" in group_df.columns:
            boundary_counts = group_df.groupby("set_boundary").size()
            row["boundary_rows_min"] = int(boundary_counts.min())
            row["boundary_rows_max"] = int(boundary_counts.max())
            row["boundary_rows_range"] = int(boundary_counts.max() - boundary_counts.min())
            row["boundary_rows_imbalanced"] = bool(boundary_counts.nunique() > 1)

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.merge(
        missing_summary,
        on=["NextWordPredictionModelRaw", "NextWordPredictionModel"],
        how="left",
    )

    for column in MISSING_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = 0
        summary[column] = summary[column].fillna(0).astype(int)

    summary["total_candidate_rows"] = summary["observed_rows"] + summary["missing_rows"]
    summary["coverage"] = np.where(
        summary["total_candidate_rows"] > 0,
        summary["observed_rows"] / summary["total_candidate_rows"],
        np.nan,
    )

    return summary


def _error_bars(summary_df: pd.DataFrame) -> np.ndarray:
    lower = summary_df["average_log_likelihood"] - summary_df["ci_lower"]
    upper = summary_df["ci_upper"] - summary_df["average_log_likelihood"]
    return np.vstack([lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)])


def _plot_title_suffix(n_bootstrap: int, ci_level: float) -> str:
    if n_bootstrap <= 0:
        return ""
    return f" ({int(round(ci_level * 100))}% context-bootstrap CI)"


def plot_avg_log_likelihood_by_model(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    n_bootstrap: int,
    ci_level: float,
) -> Path:
    x_positions = list(range(len(summary)))

    plt.figure(figsize=(9, 5))
    if n_bootstrap > 0:
        plt.errorbar(
            x_positions,
            summary["average_log_likelihood"],
            yerr=_error_bars(summary),
            fmt="none",
            ecolor="#4C78A8",
            elinewidth=1.2,
            capsize=4,
            zorder=2,
        )
    plt.scatter(
        x_positions,
        summary["average_log_likelihood"],
        s=90,
        color="#4C78A8",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    plt.xlabel("NextWordPredictionModel")
    plt.ylabel("Average log likelihood")
    plt.title("Average Log Likelihood by NextWordPredictionModel" + _plot_title_suffix(n_bootstrap, ci_level))
    plt.xticks(x_positions, summary["NextWordPredictionModel"], rotation=20, ha="right")
    _style_axes()
    plt.tight_layout()

    plot_path = out_dir / "average_log_likelihood_by_next_word_prediction_model.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def plot_avg_log_likelihood_by_structure_per_model(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    n_bootstrap: int,
    ci_level: float,
) -> List[Path]:
    models_df = (
        summary[["NextWordPredictionModelRaw", "NextWordPredictionModel"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    saved_paths: List[Path] = []
    for _, model_row in models_df.iterrows():
        model_raw = model_row["NextWordPredictionModelRaw"]
        model_label = model_row["NextWordPredictionModel"]

        model_df = summary[summary["NextWordPredictionModelRaw"] == model_raw].copy()
        x_positions = list(range(len(model_df)))

        plt.figure(figsize=(8, 5))
        if n_bootstrap > 0:
            plt.errorbar(
                x_positions,
                model_df["average_log_likelihood"],
                yerr=_error_bars(model_df),
                fmt="none",
                ecolor="#F58518",
                elinewidth=1.2,
                capsize=4,
                zorder=2,
            )
        plt.scatter(
            x_positions,
            model_df["average_log_likelihood"],
            s=90,
            color="#F58518",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        plt.xlabel("AlternativeStructure")
        plt.ylabel("Average log likelihood")
        plt.title(
            f"Average Log Likelihood by AlternativeStructure ({model_label})"
            + _plot_title_suffix(n_bootstrap, ci_level)
        )
        plt.xticks(x_positions, model_df["AlternativeStructure"], rotation=15, ha="right")
        _style_axes()
        plt.tight_layout()

        stem = _safe_stem(model_raw)
        plot_path = out_dir / f"average_log_likelihood_by_alternative_structure__{stem}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        saved_paths.append(plot_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_results_dir = root_dir / "results" / "focus_alt_exp"

    parser = argparse.ArgumentParser(description="Plot focus_alt_exp result summaries")
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plot outputs (default: <results-dir>/plots)",
    )
    parser.add_argument(
        "--structures",
        type=str,
        default=",".join(DEFAULT_STRUCTURE_ORDER),
        help="Comma-separated structures to include (default: set,ordering,conjunction,disjunction)",
    )
    parser.add_argument(
        "--model-order",
        type=str,
        default="",
        help="Optional comma-separated model order (raw names like cloze,bert,bert_static,qwen)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of context-bootstrap resamples for confidence intervals (default: 1000, use 0 to disable)",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals, expressed as a value in (0, 1).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=7,
        help="Random seed for bootstrap confidence intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    structures = _parse_csv_list(args.structures)
    if not structures:
        raise ValueError("--structures produced an empty list")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be >= 0")
    if not 0 < args.ci_level < 1:
        raise ValueError("--ci-level must be in the open interval (0, 1)")

    model_order = _parse_csv_list(args.model_order)
    out_dir = args.output_dir if args.output_dir is not None else args.results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = _collect_results(args.results_dir, structures)
    missing_trials = _collect_missing_trials(
        args.results_dir,
        model_names=all_results["NextWordPredictionModelRaw"].unique(),
    )
    missing_summary = _summarize_missing_trials(missing_trials)

    model_summary = _summarize_results(
        all_results,
        group_cols=["NextWordPredictionModelRaw", "NextWordPredictionModel"],
        missing_summary=missing_summary,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    model_summary = _apply_model_order(model_summary, model_order)

    structure_summary = _summarize_results(
        all_results,
        group_cols=["NextWordPredictionModelRaw", "NextWordPredictionModel", "AlternativeStructure"],
        missing_summary=missing_summary,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    structure_summary["AlternativeStructure"] = pd.Categorical(
        structure_summary["AlternativeStructure"],
        categories=structures,
        ordered=True,
    )
    structure_summary = _apply_model_order(
        structure_summary,
        model_order,
        extra_sort_cols=["AlternativeStructure"],
    )

    high_level_plot = plot_avg_log_likelihood_by_model(
        summary=model_summary,
        out_dir=out_dir,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.ci_level,
    )
    per_model_plots = plot_avg_log_likelihood_by_structure_per_model(
        summary=structure_summary,
        out_dir=out_dir,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.ci_level,
    )

    model_csv = out_dir / "average_log_likelihood_by_next_word_prediction_model.csv"
    model_diag_csv = out_dir / "diagnostics_by_next_word_prediction_model.csv"
    structure_diag_csv = out_dir / "diagnostics_by_model_and_alternative_structure.csv"
    missing_csv = out_dir / "missing_trials_by_next_word_prediction_model.csv"

    model_summary.to_csv(model_csv, index=False)
    model_summary.to_csv(model_diag_csv, index=False)
    structure_summary.to_csv(structure_diag_csv, index=False)
    if not missing_summary.empty:
        missing_summary.to_csv(missing_csv, index=False)

    saved_paths = [high_level_plot, model_csv, model_diag_csv, structure_diag_csv]
    if not missing_summary.empty:
        saved_paths.append(missing_csv)

    for _, model_row in (
        structure_summary[["NextWordPredictionModelRaw", "NextWordPredictionModel"]]
        .drop_duplicates()
        .iterrows()
    ):
        model_raw = model_row["NextWordPredictionModelRaw"]
        stem = _safe_stem(model_raw)
        csv_path = out_dir / f"average_log_likelihood_by_alternative_structure__{stem}.csv"
        structure_summary[structure_summary["NextWordPredictionModelRaw"] == model_raw].to_csv(
            csv_path,
            index=False,
        )
        saved_paths.append(csv_path)

    saved_paths.extend(per_model_plots)
    for path in saved_paths:
        print("Saved:", path)


if __name__ == "__main__":
    main()
