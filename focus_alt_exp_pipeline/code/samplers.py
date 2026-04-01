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

try:
    from .data_utils import normalize_unique_tokens, read_frequency_counts
except ImportError:
    from data_utils import normalize_unique_tokens, read_frequency_counts

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

    def supports_exact_model(self, model_name: str) -> bool:
        """Whether this sampler can score `model_name` without sampling."""
        del model_name
        return False

    def exact_negation_probability(
        self,
        *,
        model_name: str,
        context: str,
        query: str,
        trigger: str,
        set_boundary: int,
    ) -> float | None:
        """Return an exact negation probability when available."""
        del model_name, context, query, trigger, set_boundary
        return None

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
    """Global Google Ngram baseline shared across all experiment contexts."""

    def __init__(
        self,
        *,
        required_tokens: Sequence[str] | None = None,
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

        if required_tokens is not None and supported_tokens is not None:
            raise ValueError(
                "FrequencySampler received both required_tokens and supported_tokens; "
                "use required_tokens"
            )
        if required_tokens is None:
            required_tokens = supported_tokens

        normalized_tokens = normalize_unique_tokens(required_tokens)

        if max_vocab_size is not None:
            if max_vocab_size <= 0:
                raise ValueError("FrequencySampler max_vocab_size must be > 0")
            kept_rows = self._read_top_counts(
                unigram_counts_path=unigram_counts_path,
                bigram_counts_path=bigram_counts_path,
                limit=max_vocab_size,
            )
            if not kept_rows:
                raise ValueError("FrequencySampler could not load any top-vocab counts")
        elif background_vocab_size is not None:
            if background_vocab_size <= 0:
                raise ValueError("FrequencySampler background_vocab_size must be > 0")

            counts = self._read_combined_counts(
                unigram_counts_path=unigram_counts_path,
                bigram_counts_path=bigram_counts_path,
                required_tokens=normalized_tokens,
                background_vocab_size=background_vocab_size,
            )
            if not counts:
                raise ValueError("FrequencySampler could not load any positive token counts")
            kept_rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        else:
            if not normalized_tokens:
                raise ValueError(
                    "FrequencySampler requires required_tokens when max_vocab_size is not set"
                )

            counts = read_frequency_counts(
                normalized_tokens,
                unigram_counts_path=unigram_counts_path,
                bigram_counts_path=bigram_counts_path,
            )
            kept_rows = [
                (token, counts[token])
                for token in normalized_tokens
                if counts.get(token, 0) > 0
            ]
            if not kept_rows:
                raise ValueError("FrequencySampler could not load any positive token counts")

        kept_tokens = [token for token, _ in kept_rows]
        probs = np.array([float(count) for _, count in kept_rows], dtype=float)

        probs = probs / probs.sum()
        self._tokens = np.array(kept_tokens, dtype=object)
        self._probs = probs
        self._token_set = set(kept_tokens)
        self._token_counts = {token: int(count) for token, count in kept_rows}

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
        required_tokens: Sequence[str],
        background_vocab_size: int,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token, count in cls._read_top_counts(
            unigram_counts_path=unigram_counts_path,
            bigram_counts_path=bigram_counts_path,
            limit=background_vocab_size,
        ):
            counts[token] = count

        remaining_tokens = [token for token in required_tokens if token not in counts]
        if remaining_tokens:
            counts.update(
                read_frequency_counts(
                    remaining_tokens,
                    unigram_counts_path=unigram_counts_path,
                    bigram_counts_path=bigram_counts_path,
                )
            )

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

    def supports_exact_model(self, model_name: str) -> bool:
        return model_name == "ordering"

    def exact_negation_probability(
        self,
        *,
        model_name: str,
        context: str,
        query: str,
        trigger: str,
        set_boundary: int,
    ) -> float | None:
        del context, set_boundary
        if model_name != "ordering":
            return None

        query_key = str(query).strip().lower()
        trigger_key = str(trigger).strip().lower()
        if query_key == trigger_key:
            return 0.0
        query_count = self._token_counts.get(query_key)
        trigger_count = self._token_counts.get(trigger_key)
        if query_count is None or trigger_count is None:
            return None
        return float(query_count) / float(query_count + trigger_count)


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
