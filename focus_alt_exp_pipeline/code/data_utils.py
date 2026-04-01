"""Shared token normalization and experiment-data helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


def clean_word(word: str) -> str:
    token = str(word).strip().lower()
    if token.startswith("a "):
        return token[2:]
    if token.startswith("an "):
        return token[3:]
    return token


def prepare_experimental_data(experimental_data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized cleaned query/trigger columns."""
    df = experimental_data.copy()
    if "cleaned_query" not in df.columns and "query" in df.columns:
        df["cleaned_query"] = df["query"].apply(clean_word)
    if "cleaned_trigger" not in df.columns and "trigger" in df.columns:
        df["cleaned_trigger"] = df["trigger"].apply(clean_word)
    if "story" in df.columns:
        df = df.sort_values(by="story")
    return df


def resolve_context_col(df: pd.DataFrame) -> str:
    if "story" in df.columns:
        return "story"
    if "context" in df.columns:
        return "context"
    raise KeyError("Expected either a 'story' or 'context' column in experimental data")


def normalize_unique_tokens(values: Sequence[str] | None) -> List[str]:
    if values is None:
        return []

    normalized: List[str] = []
    seen = set()
    for value in values:
        token = clean_word(value)
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def read_token_counts(path: str | Path, targets: Iterable[str]) -> Dict[str, int]:
    normalized_targets = normalize_unique_tokens(list(targets))
    remaining = set(normalized_targets)
    counts: Dict[str, int] = {}
    if not remaining:
        return counts

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            token, sep, count_str = raw_line.rstrip("\n").partition("\t")
            if not sep or token not in remaining:
                continue
            counts[token] = int(count_str)
            remaining.remove(token)
            if not remaining:
                break
    return counts


def read_frequency_counts(
    tokens: Sequence[str],
    *,
    unigram_counts_path: str | Path,
    bigram_counts_path: str | Path,
) -> Dict[str, int]:
    normalized = normalize_unique_tokens(tokens)
    unigram_targets = {token for token in normalized if " " not in token}
    bigram_targets = {token for token in normalized if " " in token}

    counts: Dict[str, int] = {}
    if unigram_targets:
        counts.update(read_token_counts(unigram_counts_path, unigram_targets))
    if bigram_targets:
        counts.update(read_token_counts(bigram_counts_path, bigram_targets))
    return counts
