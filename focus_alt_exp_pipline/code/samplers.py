"""Dataset-specific sample generation.

This module owns one responsibility:
given contexts + set boundary, generate sampled orderings/sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

Sample = Tuple[List[str], List[str]]
ContextSamples = Dict[str, List[Sample]]


class ContextSampler(ABC):
    """Interface for dataset-specific context sampling."""

    def prepare_contexts(self, contexts: Sequence[str]) -> None:
        """Optional one-time precompute hook before set-boundary loops."""
        del contexts

    def allow_unsupported_query_for_model(self, model_name: str) -> bool:
        """Whether a missing query should still be scored for a specific model."""
        del model_name
        return False

    @abstractmethod
    def sample_contexts(
        self,
        contexts: Sequence[str],
        set_boundary: int,
        num_reps: int,
    ) -> ContextSamples:
        """Return context -> list[(sampled_ordering, sampled_set)]."""

    @abstractmethod
    def supports_token(self, context: str, token: str) -> bool:
        """True if `token` is available for this context's distribution."""


class FrequencySampler(ContextSampler):
    """Context-free sampler backed by unigram and bigram frequency counts."""

    def __init__(
        self,
        *,
        supported_tokens: Sequence[str] | None = None,
        unigram_counts_path: str | Path,
        bigram_counts_path: str | Path,
        background_vocab_size: int | None = None,
        max_vocab_size: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._background_vocab_size = background_vocab_size
        self._max_vocab_size = max_vocab_size

        if background_vocab_size is not None and max_vocab_size is not None:
            raise ValueError(
                "FrequencySampler background_vocab_size and max_vocab_size cannot both be set"
            )

        normalized_tokens = self._normalize_tokens(supported_tokens)

        if max_vocab_size is not None:
            if max_vocab_size <= 0:
                raise ValueError("FrequencySampler max_vocab_size must be > 0")
            top_rows = self._read_top_counts(
                unigram_counts_path=unigram_counts_path,
                bigram_counts_path=bigram_counts_path,
                limit=max_vocab_size,
            )
            if not top_rows:
                raise ValueError("FrequencySampler could not load any top-vocab counts")
            kept_tokens = [token for token, _ in top_rows]
            probs = np.array([float(count) for _, count in top_rows], dtype=float)
        elif background_vocab_size is not None:
            if background_vocab_size <= 0:
                raise ValueError("FrequencySampler background_vocab_size must be > 0")

            counts = self._read_combined_counts(
                unigram_counts_path=unigram_counts_path,
                bigram_counts_path=bigram_counts_path,
                supported_tokens=normalized_tokens,
                background_vocab_size=background_vocab_size,
            )
            if not counts:
                raise ValueError("FrequencySampler could not load any positive token counts")
            kept_rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            kept_tokens = [token for token, _ in kept_rows]
            probs = np.array([float(count) for _, count in kept_rows], dtype=float)
        else:
            if not normalized_tokens:
                raise ValueError("FrequencySampler requires supported_tokens when max_vocab_size is not set")

            unigram_targets = {token for token in normalized_tokens if " " not in token}
            bigram_targets = {token for token in normalized_tokens if " " in token}

            counts: Dict[str, int] = {}
            if unigram_targets:
                counts.update(self._read_counts(unigram_counts_path, unigram_targets))
            if bigram_targets:
                counts.update(self._read_counts(bigram_counts_path, bigram_targets))

            kept_tokens = [token for token in normalized_tokens if counts.get(token, 0) > 0]
            if not kept_tokens:
                raise ValueError("FrequencySampler could not load any positive token counts")
            probs = np.array([float(counts[token]) for token in kept_tokens], dtype=float)

        probs = probs / probs.sum()
        self._tokens = np.array(kept_tokens, dtype=object)
        self._probs = probs
        self._token_set = set(kept_tokens)

    @staticmethod
    def _normalize_tokens(tokens: Sequence[str] | None) -> List[str]:
        if tokens is None:
            return []

        normalized: List[str] = []
        seen = set()
        for token in tokens:
            key = str(token).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        return normalized

    @staticmethod
    def _read_counts(path: str | Path, targets: set[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        remaining = set(targets)

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

    @staticmethod
    def _read_top_counts(
        *,
        unigram_counts_path: str | Path,
        bigram_counts_path: str | Path,
        limit: int,
    ) -> List[Tuple[str, int]]:
        rows: List[Tuple[str, int]] = []
        for path in [unigram_counts_path, bigram_counts_path]:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    if line_no > limit:
                        break
                    token, sep, count_str = raw_line.rstrip("\n").partition("\t")
                    if not sep:
                        continue
                    rows.append((token, int(count_str)))
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows[:limit]

    @classmethod
    def _read_combined_counts(
        cls,
        *,
        unigram_counts_path: str | Path,
        bigram_counts_path: str | Path,
        supported_tokens: Sequence[str],
        background_vocab_size: int,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token, count in cls._read_top_counts(
            unigram_counts_path=unigram_counts_path,
            bigram_counts_path=bigram_counts_path,
            limit=background_vocab_size,
        ):
            counts[token] = count

        remaining_unigrams = {
            token for token in supported_tokens if " " not in token and token not in counts
        }
        remaining_bigrams = {
            token for token in supported_tokens if " " in token and token not in counts
        }
        if remaining_unigrams:
            counts.update(cls._read_counts(unigram_counts_path, remaining_unigrams))
        if remaining_bigrams:
            counts.update(cls._read_counts(bigram_counts_path, remaining_bigrams))

        return {token: count for token, count in counts.items() if count > 0}

    def _sample_ordering(self) -> List[str]:
        sampled_idx = self._rng.choice(
            len(self._tokens),
            size=len(self._tokens),
            replace=False,
            p=self._probs,
        )
        return self._tokens[sampled_idx].tolist()

    def prepare_contexts(self, contexts: Sequence[str]) -> None:
        del contexts

    def sample_contexts(
        self,
        contexts: Sequence[str],
        set_boundary: int,
        num_reps: int,
    ) -> ContextSamples:
        context_samples: ContextSamples = {}
        for context in contexts:
            key = str(context)
            sims: List[Sample] = []
            for _ in range(num_reps):
                ordering = self._sample_ordering()
                sims.append((ordering, ordering[:set_boundary]))
            context_samples[key] = sims
        return context_samples

    def supports_token(self, context: str, token: str) -> bool:
        del context
        return str(token).strip().lower() in self._token_set

    def allow_unsupported_query_for_model(self, model_name: str) -> bool:
        return self._max_vocab_size is not None and model_name == "set"


class ClozeSampler(ContextSampler):
    """Sampler for cloze-probability distributions."""

    def __init__(
        self,
        cloze_df: pd.DataFrame,
        *,
        context_col: str = "context",
        word_col: str = "word",
        prob_col: str = "cloze_probability",
        seed: int | None = None,
    ) -> None:
        self._context_col = context_col
        self._word_col = word_col
        self._prob_col = prob_col
        self._rng = np.random.default_rng(seed)
        self._context_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._context_words: Dict[str, set[str]] = {}

        required = {context_col, word_col, prob_col}
        missing = required.difference(cloze_df.columns)
        if missing:
            raise ValueError(f"Missing columns in cloze_df: {sorted(missing)}")

        for context, subset in cloze_df.groupby(context_col, sort=False):
            words = subset[word_col].astype(str).to_numpy()
            probs = subset[prob_col].to_numpy(dtype=float)
            if len(words) == 0:
                continue
            prob_sum = probs.sum()
            if prob_sum <= 0:
                raise ValueError(f"Invalid probabilities for context '{context}'")
            probs = probs / prob_sum
            self._context_data[str(context)] = (words, probs)
            self._context_words[str(context)] = set(words.tolist())

    def _sample_ordering(self, context: str) -> List[str]:
        words, probs = self._context_data[context]
        positive_idx = np.where(probs > 0)[0]
        zero_idx = np.where(probs <= 0)[0]

        if len(positive_idx) == 0:
            order = self._rng.permutation(len(words))
            return words[order].tolist()

        pos_probs = probs[positive_idx]
        pos_probs = pos_probs / pos_probs.sum()
        sampled_positive = self._rng.choice(
            positive_idx,
            size=len(positive_idx),
            replace=False,
            p=pos_probs,
        )

        if len(zero_idx) > 0:
            sampled_zero = self._rng.permutation(zero_idx)
            final_idx = np.concatenate([sampled_positive, sampled_zero])
        else:
            final_idx = sampled_positive

        return words[final_idx].tolist()

    def sample_contexts(
        self,
        contexts: Sequence[str],
        set_boundary: int,
        num_reps: int,
    ) -> ContextSamples:
        context_samples: ContextSamples = {}
        for context in contexts:
            key = str(context)
            if key not in self._context_data:
                raise KeyError(f"No cloze distribution for context '{key}'")
            sims: List[Sample] = []
            for _ in range(num_reps):
                ordering = self._sample_ordering(key)
                sims.append((ordering, ordering[:set_boundary]))
            context_samples[key] = sims
        return context_samples

    def supports_token(self, context: str, token: str) -> bool:
        key = str(context)
        return token in self._context_words.get(key, set())
