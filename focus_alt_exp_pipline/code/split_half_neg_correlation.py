from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/"
    "focus_alt_exp_pipline/human_exp_data/sca_dataframe.csv"
)
DEFAULT_OUTPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/"
    "focus_alt_exp_pipline/results/split_half_neg_proportions.csv"
)
GROUP_COLUMNS = ["story", "trigger", "query"]
REQUIRED_COLUMNS = GROUP_COLUMNS + ["neg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split neg responses into two random halves within each "
            "story/trigger/query group and correlate half-level proportions."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the participant-level CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the split-half summary CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="Random seed for the within-group split. Default: 200",
    )
    parser.add_argument(
        "--method",
        choices=["pearson"],
        default="pearson",
        help="Correlation method to report. Default: pearson",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def coerce_neg_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neg"] = pd.to_numeric(df["neg"], errors="raise")

    invalid = df.loc[~df["neg"].isin([0, 1]), REQUIRED_COLUMNS]
    if not invalid.empty:
        raise ValueError(
            "The 'neg' column must contain only 0/1 values. "
            f"Found invalid rows:\n{invalid.head().to_string(index=False)}"
        )

    df["neg"] = df["neg"].astype(int)
    return df


def split_half_proportions(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    group_sizes = df.groupby(GROUP_COLUMNS).size()
    too_small = group_sizes[group_sizes < 2]
    if not too_small.empty:
        raise ValueError(
            "Each story/trigger/query group must have at least 2 observations. "
            f"Found undersized groups:\n{too_small.head().to_string()}"
        )

    for group_key, group_df in df.groupby(GROUP_COLUMNS, sort=True):
        neg_values = group_df["neg"].to_numpy()
        shuffled = rng.permutation(neg_values)
        half_1, half_2 = np.array_split(shuffled, 2)

        records.append(
            {
                "story": group_key[0],
                "trigger": group_key[1],
                "query": group_key[2],
                "neg_proportion_H1": float(half_1.mean()),
                "neg_proportion_H2": float(half_2.mean()),
            }
        )

    return pd.DataFrame.from_records(records).sort_values(GROUP_COLUMNS).reset_index(drop=True)


def compute_correlation(df: pd.DataFrame, method: str) -> float:
    x = df["neg_proportion_H1"].to_numpy()
    y = df["neg_proportion_H2"].to_numpy()

    if method == "pearson":
        corr_x = x
        corr_y = y
    else:
        corr_x = pd.Series(x).rank(method="average").to_numpy()
        corr_y = pd.Series(y).rank(method="average").to_numpy()

    if np.std(corr_x) == 0 or np.std(corr_y) == 0:
        return float("nan")

    correlation = np.corrcoef(corr_x, corr_y)[0, 1]
    return float(correlation)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    validate_columns(df)
    df = coerce_neg_column(df)

    summary_df = split_half_proportions(df, seed=args.seed)
    correlation = compute_correlation(summary_df, method=args.method)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Grouped rows: {len(summary_df)}")
    print(f"Output CSV: {args.output}")
    print(
        f"{args.method.title()} correlation between neg_proportion_H1 and "
        f"neg_proportion_H2: {correlation:.6f}"
    )


if __name__ == "__main__":
    main()
