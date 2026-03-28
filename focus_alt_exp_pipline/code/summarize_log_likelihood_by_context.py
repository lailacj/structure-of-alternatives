"""Summarize average log likelihood by context and alternative structure.

This script reads raw focus_alt_exp result CSVs such as
``ordering_results_cloze.csv`` and ``set_results_frequency.csv`` and produces
per-context summary tables for the requested models and structures.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd


FILE_PATTERN = re.compile(r"^(set|ordering|conjunction|disjunction)_results_(.+)\.csv$")
DEFAULT_STRUCTURE_FILTER = ["ordering", "set", "conjunction", "disjunction"]
DEFAULT_MODEL_FILTER = ["cloze", "frequency"]


def _parse_csv_list(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _pretty_model_name(raw_name: str) -> str:
    acronym_map = {
        "frequency": "Frequency",
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


def _collect_results(
    results_dir: Path,
    structures: Iterable[str],
    model_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    structure_set = set(structures)
    model_set = set(model_names) if model_names else None
    frames = []

    for path in sorted(results_dir.glob("*_results_*.csv")):
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


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    default_results_dir = root_dir / "focus_alt_exp_pipline" / "results"

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
    summary = summary.sort_values(
        ["_model_order", "NextWordPredictionModel", "AlternativeStructure", "context"],
        ignore_index=True,
    ).drop(columns="_model_order")

    combined_csv = out_dir / "average_log_likelihood_by_context_and_alternative_structure.csv"
    summary.to_csv(combined_csv, index=False)
    print("Saved:", combined_csv)

    for model_raw in summary["NextWordPredictionModelRaw"].drop_duplicates():
        model_csv = out_dir / (
            f"average_log_likelihood_by_context_and_alternative_structure__{_safe_stem(model_raw)}.csv"
        )
        summary[summary["NextWordPredictionModelRaw"] == model_raw].to_csv(model_csv, index=False)
        print("Saved:", model_csv)


if __name__ == "__main__":
    main()
