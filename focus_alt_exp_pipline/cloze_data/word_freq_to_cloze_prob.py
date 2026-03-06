"""
Build one cloze CSV from two raw generation CSV files.

Output columns:
  context, word, frequency, cloze_probability

Rules:
  - For positive=True rows, cloze_probability = frequency / n_positive_participants.
  - For positive=False rows, start with frequency / n_negative_participants, then scale
    probabilities per context so all negatives are strictly below the minimum positive
    cloze probability for that context, while reversing negative rank/order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_A = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/next_word_prediction_correlations/data/inside_the_set/Generative_Data_RAW.csv"
)
DEFAULT_INPUT_B = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipline/cloze_data/generative_data_raw_pt2.csv"
)
DEFAULT_OUTPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipline/cloze_data/all_cloze_prob_data.csv"
)

RESPONSES_PER_CONTEXT = 6
EPSILON = 1e-12

TRUE_VALUES = {"true", "1", "t", "yes", "y"}
MISSING_TOKENS = {"", "nan", "none", "na", "n/a", "x"}

# raw context name -> output context name
CONTEXT_MAP = {
    "handbag": "bag",
    "bakery": "bakery",
    "closet": "beach",
    "cold": "cold",
    "cut": "cut",
    "fridge": "fridge",
    "fitness": "gym",
    "hot": "hot",
    "mall": "mall",
    "mask": "mask",
    "meat": "meat",
    "restaurant": "restaurant",
    "corner": "salad",
    "library": "science",
    "garage": "throw",
    "transport": "transport",
}


def parse_bool_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def find_context_columns(df: pd.DataFrame, raw_context: str) -> list[str] | None:
    plain = [f"{raw_context}{i}" for i in range(1, RESPONSES_PER_CONTEXT + 1)]
    underscored = [f"{raw_context}_{i}" for i in range(1, RESPONSES_PER_CONTEXT + 1)]

    if all(col in df.columns for col in plain):
        return plain
    if all(col in df.columns for col in underscored):
        return underscored
    return None


def count_words(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = []
    for col in columns:
        cleaned = df[col].dropna().astype(str).str.strip().str.casefold()
        parts.append(cleaned)

    if not parts:
        return pd.Series(dtype="int64")

    words = pd.concat(parts, ignore_index=True)
    words = words[~words.isin(MISSING_TOKENS)]
    words = words[words != ""]
    if words.empty:
        return pd.Series(dtype="int64")

    return words.value_counts().astype(int)


def build_rows_for_context(
    df: pd.DataFrame,
    raw_context: str,
    output_context: str,
) -> pd.DataFrame:
    columns = find_context_columns(df, raw_context)
    if columns is None:
        return pd.DataFrame(columns=["context", "word", "frequency", "cloze_probability", "condition"])

    if "positive" not in df.columns:
        raise ValueError("Each input CSV must include a 'positive' column.")

    pos_mask = parse_bool_true(df["positive"])
    pos_df = df.loc[pos_mask]
    neg_df = df.loc[~pos_mask]

    n_positive = int(pos_mask.sum())
    n_negative = int((~pos_mask).sum())

    if n_positive == 0:
        raise ValueError(f"No positive=True rows found for context '{raw_context}'.")

    rows = []

    # Positive rows.
    pos_counts = count_words(pos_df, columns)
    for word, frequency in pos_counts.items():
        rows.append(
            {
                "context": output_context,
                "word": word,
                "frequency": int(frequency),
                "cloze_probability": float(frequency / n_positive),
                "condition": "positive",
            }
        )

    if pos_counts.empty or n_negative == 0:
        return pd.DataFrame(rows)

    # Negative rows: reverse raw rank/order, then keep all values under min positive cloze.
    neg_counts = count_words(neg_df, columns)
    if neg_counts.empty:
        return pd.DataFrame(rows)

    min_positive = float((pos_counts / n_positive).min())
    neg_base_probs = neg_counts / n_negative
    max_neg_base = float(neg_base_probs.max())
    cap = max(min_positive - EPSILON, 0.0)

    if max_neg_base <= 0.0 or cap <= 0.0:
        scaled_neg_probs = neg_base_probs * 0.0
    else:
        # Flip negatives so the highest base probability becomes the lowest scaled probability.
        flipped = 1.0 - (neg_base_probs / max_neg_base)
        scaled_neg_probs = flipped.clip(lower=0.0) * cap

    for word, frequency in neg_counts.items():
        rows.append(
            {
                "context": output_context,
                "word": word,
                "frequency": int(frequency),
                "cloze_probability": float(scaled_neg_probs[word]),
                "condition": "negative",
            }
        )

    return pd.DataFrame(rows)


def drop_negative_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    positive_pairs = (
        df.loc[df["condition"] == "positive", ["context", "word"]]
        .drop_duplicates()
        .assign(in_positive=True)
    )
    merged = df.merge(positive_pairs, on=["context", "word"], how="left")
    keep = ~((merged["condition"] == "negative") & (merged["in_positive"] == True))
    return merged.loc[keep, ["context", "word", "frequency", "cloze_probability", "condition"]].copy()


def build_output(input_paths: list[Path], drop_overlap_negatives: bool) -> pd.DataFrame:
    frames = []
    found_contexts: set[str] = set()

    for path in input_paths:
        df = pd.read_csv(path)
        for raw_context, output_context in CONTEXT_MAP.items():
            frame = build_rows_for_context(df, raw_context, output_context)
            if frame.empty:
                continue
            frames.append(frame)
            found_contexts.add(output_context)

    missing_contexts = sorted(set(CONTEXT_MAP.values()) - found_contexts)
    if missing_contexts:
        raise ValueError(f"Could not find these required contexts in the input files: {missing_contexts}")

    output = pd.concat(frames, ignore_index=True)
    if drop_overlap_negatives:
        output = drop_negative_overlaps(output)

    output = output.drop(columns=["condition"])
    output = output.sort_values(
        ["context", "cloze_probability", "frequency", "word"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two raw generation CSVs into one context/word/frequency/cloze table."
    )
    parser.add_argument("--input-a", type=Path, default=DEFAULT_INPUT_A)
    parser.add_argument("--input-b", type=Path, default=DEFAULT_INPUT_B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--keep-overlap-negatives",
        action="store_true",
        help="Keep negative rows even if the same (context, word) appears in positives.",
    )
    args = parser.parse_args()

    output_df = build_output(
        input_paths=[args.input_a, args.input_b],
        drop_overlap_negatives=not args.keep_overlap_negatives,
    )
    output_df.to_csv(args.output, index=False)
    print(f"[done] Wrote {len(output_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
