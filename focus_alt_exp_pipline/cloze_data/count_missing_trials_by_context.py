#!/usr/bin/env python3
"""Count missing trials per context for a selected set boundary."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def normalize_set_boundary(value: str) -> str:
    """Normalize CSV set-boundary values to an integer-like string."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def validate_columns(path: Path, fieldnames: list[str], required: set[str]) -> None:
    missing = required.difference(fieldnames)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_list}")


def load_human_counts(human_csv: Path) -> Counter:
    counts: Counter = Counter()
    with human_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{human_csv} has no header row")
        validate_columns(human_csv, reader.fieldnames, {"story"})

        for row in reader:
            context = row["story"].strip()
            counts[context] += 1
    return counts


def load_missing_counts(missing_csv: Path, set_boundary: int) -> Counter:
    counts: Counter = Counter()
    target = str(set_boundary)

    with missing_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{missing_csv} has no header row")
        validate_columns(missing_csv, reader.fieldnames, {"set_boundary", "context"})

        for row in reader:
            boundary = normalize_set_boundary(row["set_boundary"])
            if boundary != target:
                continue
            context = row["context"].strip()
            counts[context] += 1
    return counts


def print_summary(human_counts: Counter, missing_counts: Counter, set_boundary: int) -> None:
    contexts = sorted(human_counts)

    header = (
        f"Missing-trial summary by context (set_boundary={set_boundary})\n"
        f"{'context':<14} {'human_rows':>10} {'missing_rows':>12} "
        f"{'available_rows':>14} {'missing_pct':>12}"
    )
    print(header)
    print("-" * len(header.splitlines()[-1]))

    total_human = 0
    total_missing_on_human = 0

    for context in contexts:
        human_rows = human_counts[context]
        missing_rows = missing_counts.get(context, 0)
        available_rows = human_rows - missing_rows
        missing_pct = (missing_rows / human_rows * 100.0) if human_rows else 0.0

        total_human += human_rows
        total_missing_on_human += missing_rows

        print(
            f"{context:<14} {human_rows:>10} {missing_rows:>12} "
            f"{available_rows:>14} {missing_pct:>11.1f}%"
        )

    print("-" * len(header.splitlines()[-1]))
    print(
        f"{'TOTAL (human contexts)':<14} {total_human:>10} {total_missing_on_human:>12} "
        f"{(total_human - total_missing_on_human):>14} "
        f"{(total_missing_on_human / total_human * 100.0):>11.1f}%"
    )

    extra_contexts = sorted(set(missing_counts).difference(human_counts))
    if extra_contexts:
        extra_missing = sum(missing_counts[c] for c in extra_contexts)
        print("\nMissing rows found for contexts not present in human data:")
        for context in extra_contexts:
            print(f"- {context}: {missing_counts[context]}")
        print(f"Total missing rows in extra contexts: {extra_missing}")


def write_output_csv(
    out_csv: Path,
    human_counts: Counter,
    missing_counts: Counter,
    set_boundary: int,
) -> None:
    contexts = sorted(set(human_counts) | set(missing_counts))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set_boundary",
                "context",
                "human_rows",
                "missing_rows",
                "available_rows",
                "missing_pct",
            ]
        )

        for context in contexts:
            human_rows = human_counts.get(context, 0)
            missing_rows = missing_counts.get(context, 0)
            available_rows = human_rows - missing_rows
            missing_pct = (missing_rows / human_rows * 100.0) if human_rows else 0.0
            writer.writerow(
                [
                    set_boundary,
                    context,
                    human_rows,
                    missing_rows,
                    available_rows,
                    round(missing_pct, 4),
                ]
            )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_human = project_root / "human_exp_data" / "sca_dataframe.csv"
    default_missing = project_root / "results" / "missing_trials_cloze.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Count human rows and missing rows per context for a chosen set boundary."
        )
    )
    parser.add_argument(
        "--set-boundary",
        type=int,
        default=3,
        help="Set boundary value to filter missing trials (default: 3).",
    )
    parser.add_argument(
        "--human-csv",
        type=Path,
        default=default_human,
        help=f"Path to human data CSV (default: {default_human}).",
    )
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=default_missing,
        help=f"Path to missing-trials CSV (default: {default_missing}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default= project_root / "results" / "missing_trials_summary.csv",
        help="Optional output path for the summary as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_counts = load_human_counts(args.human_csv)
    missing_counts = load_missing_counts(args.missing_csv, args.set_boundary)

    print_summary(human_counts, missing_counts, args.set_boundary)

    if args.out_csv is not None:
        write_output_csv(args.out_csv, human_counts, missing_counts, args.set_boundary)
        print(f"\nWrote CSV summary to: {args.out_csv}")


if __name__ == "__main__":
    main()
