"""Dataset-specific sample generation.

This module owns one responsibility:
given contexts + set boundary, generate sampled orderings/sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .data_utils import normalize_unique_tokens, read_frequency_counts
    from .models import ModelResult, probability_to_model_result
except ImportError:
    from data_utils import normalize_unique_tokens, read_frequency_counts
    from models import ModelResult, probability_to_model_result

Sample = Tuple[List[str], List[str]]
ContextSamples = Dict[str, List[Sample]]


def _sample_weighted_indices(
    rng: np.random.Generator,
    probs: np.ndarray,
    *,
    sample_size: int | None = None,
) -> np.ndarray:
    total_size = len(probs)
    if sample_size is None:
        sample_size = total_size
    sample_size = min(int(sample_size), total_size)
    if sample_size <= 0:
        return np.empty(0, dtype=np.int64)

    positive_idx = np.flatnonzero(probs > 0)
    zero_idx = np.flatnonzero(probs <= 0)

    if len(positive_idx) == 0:
        return rng.permutation(total_size)[:sample_size].astype(np.int64)

    positive_sample_size = min(sample_size, len(positive_idx))
    positive_probs = np.asarray(probs[positive_idx], dtype=np.float64)
    positive_probs = positive_probs / positive_probs.sum()
    sampled_positive = rng.choice(
        positive_idx,
        size=positive_sample_size,
        replace=False,
        p=positive_probs,
    ).astype(np.int64)

    if positive_sample_size == sample_size:
        return sampled_positive

    remaining = sample_size - positive_sample_size
    sampled_zero = rng.permutation(zero_idx)[:remaining].astype(np.int64)
    return np.concatenate([sampled_positive, sampled_zero]).astype(np.int64)


class ContextSampler(ABC):
    """Interface for dataset-specific context sampling."""

    def prepare_contexts(self, contexts: Sequence[str]) -> None:
        """Optional one-time precompute hook before set-boundary loops."""
        del contexts

    def prepare_run(
        self,
        *,
        contexts: Sequence[str],
        set_boundaries: Sequence[int],
        num_reps: int,
        model_names: Sequence[str],
    ) -> None:
        """Optional hook to precompute run-level state once."""
        del contexts, set_boundaries, num_reps, model_names

    def allow_unsupported_query_for_model(self, model_name: str) -> bool:
        """Whether a missing query should still be scored for a specific model."""
        del model_name
        return False

    def supports_direct_model(self, model_name: str) -> bool:
        """Whether this sampler can score `model_name` directly."""
        del model_name
        return False

    def direct_model_result(
        self,
        *,
        model_name: str,
        context: str,
        query: str,
        trigger: str,
        query_negated: int,
        set_boundary: int,
    ) -> ModelResult | None:
        """Return a direct model result when the sampler can score it itself."""
        del model_name, context, query, trigger, query_negated, set_boundary
        return None

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
        keep_zero_count_tokens: bool = False,
        empirical_ordering: bool = False,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._background_vocab_size = background_vocab_size
        self._max_vocab_size = max_vocab_size
        self._keep_zero_count_tokens = bool(keep_zero_count_tokens)
        self._empirical_ordering = bool(empirical_ordering)

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
            kept_rows = []
            for token in normalized_tokens:
                count = counts.get(token, 0)
                if count > 0 or self._keep_zero_count_tokens:
                    kept_rows.append((token, count))
            if not any(count > 0 for _, count in kept_rows):
                raise ValueError("FrequencySampler could not load any positive token counts")
            if not kept_rows:
                raise ValueError("FrequencySampler could not load any supported tokens")

        kept_tokens = [token for token, _ in kept_rows]
        probs = np.array([float(count) for _, count in kept_rows], dtype=float)

        probs = probs / probs.sum()
        self._tokens = np.array(kept_tokens, dtype=object)
        self._probs = probs
        self._token_set = set(kept_tokens)
        self._token_counts = {token: int(count) for token, count in kept_rows}
        self._required_tokens = list(normalized_tokens)
        self._required_token_to_index = {
            token: idx for idx, token in enumerate(self._required_tokens)
        }
        self._support_index_to_required_index = np.full(len(self._tokens), -1, dtype=np.int32)
        for support_idx, token in enumerate(kept_tokens):
            required_idx = self._required_token_to_index.get(token)
            if required_idx is not None:
                self._support_index_to_required_index[support_idx] = required_idx
        self._cached_num_reps: int | None = None
        self._cached_max_boundary: int | None = None
        self._cached_position_sentinel: int | None = None
        self._cached_required_positions: np.ndarray | None = None
        self._cached_empirical_num_reps: int | None = None
        self._cached_empirical_position_sentinel: int | None = None
        self._cached_empirical_required_positions: np.ndarray | None = None
        self._cached_probability_lookup: Dict[Tuple[str, str, str, int], float] = {}

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
        sampled_idx = _sample_weighted_indices(self._rng, self._probs)
        return self._tokens[sampled_idx].tolist()

    def _sample_prefix_indices(self, prefix_size: int) -> np.ndarray:
        return _sample_weighted_indices(
            self._rng,
            self._probs,
            sample_size=prefix_size,
        )

    def _ensure_global_prefix_cache(
        self,
        *,
        num_reps: int,
        max_boundary: int,
    ) -> None:
        if max_boundary <= 0:
            raise ValueError("FrequencySampler max_boundary must be > 0")

        cache_boundary = min(max_boundary, len(self._tokens))
        if (
            self._cached_num_reps == num_reps
            and self._cached_max_boundary == cache_boundary
            and self._cached_required_positions is not None
        ):
            return

        sentinel = cache_boundary
        dtype = np.int16 if sentinel <= np.iinfo(np.int16).max else np.int32
        required_positions = np.full(
            (num_reps, len(self._required_tokens)),
            sentinel,
            dtype=dtype,
        )

        for rep_idx in range(num_reps):
            sampled_idx = self._sample_prefix_indices(cache_boundary)
            sampled_tokens = self._tokens[sampled_idx].tolist()
            for position, token in enumerate(sampled_tokens):
                required_idx = self._required_token_to_index.get(str(token))
                if required_idx is not None:
                    required_positions[rep_idx, required_idx] = position

        self._cached_num_reps = num_reps
        self._cached_max_boundary = cache_boundary
        self._cached_position_sentinel = sentinel
        self._cached_required_positions = required_positions
        self._cached_probability_lookup = {}

    def _ensure_global_empirical_cache(
        self,
        *,
        num_reps: int,
    ) -> None:
        sentinel = len(self._tokens)
        if (
            self._cached_empirical_num_reps == num_reps
            and self._cached_empirical_required_positions is not None
        ):
            return

        dtype = np.int16 if sentinel <= np.iinfo(np.int16).max else np.int32
        required_positions = np.full(
            (num_reps, len(self._required_tokens)),
            sentinel,
            dtype=dtype,
        )

        for rep_idx in range(num_reps):
            sampled_idx = _sample_weighted_indices(self._rng, self._probs)
            for position, support_idx in enumerate(sampled_idx):
                required_idx = int(self._support_index_to_required_index[int(support_idx)])
                if required_idx >= 0:
                    required_positions[rep_idx, required_idx] = position

        self._cached_empirical_num_reps = num_reps
        self._cached_empirical_position_sentinel = sentinel
        self._cached_empirical_required_positions = required_positions
        self._cached_probability_lookup = {}

    def _get_required_positions(self, token: str) -> np.ndarray | None:
        positions = (
            self._cached_empirical_required_positions
            if self._empirical_ordering
            else self._cached_required_positions
        )
        if positions is None:
            return None

        token_key = str(token).strip().lower()
        token_idx = self._required_token_to_index.get(token_key)
        if token_idx is None:
            return None
        return positions[:, token_idx]

    def _estimate_set_probability(self, query: str, set_boundary: int) -> float:
        query_positions = self._get_required_positions(query)
        if query_positions is None:
            return 0.0
        if str(query).strip().lower() not in self._token_set:
            return 0.0

        if self._empirical_ordering:
            if self._cached_empirical_position_sentinel is None:
                raise RuntimeError("FrequencySampler empirical cache is not prepared")
            boundary = min(set_boundary, self._cached_empirical_position_sentinel)
        else:
            if self._cached_required_positions is None or self._cached_max_boundary is None:
                raise RuntimeError("FrequencySampler prefix cache is not prepared")
            boundary = min(set_boundary, self._cached_max_boundary)
        return float(np.mean(query_positions < boundary))

    def _estimate_conjunction_probability(
        self,
        query: str,
        trigger: str,
        set_boundary: int,
    ) -> float:
        query_positions = self._get_required_positions(query)
        if query_positions is None:
            return 0.0

        trigger_positions = self._get_required_positions(trigger)
        if trigger_positions is None:
            if self._empirical_ordering:
                if self._cached_empirical_position_sentinel is None:
                    raise RuntimeError("FrequencySampler empirical cache is not prepared")
                trigger_positions = np.full_like(
                    query_positions,
                    self._cached_empirical_position_sentinel,
                )
            else:
                if self._cached_position_sentinel is None:
                    raise RuntimeError("FrequencySampler prefix cache is not prepared")
                trigger_positions = np.full_like(query_positions, self._cached_position_sentinel)

        if self._empirical_ordering:
            if self._cached_empirical_position_sentinel is None:
                raise RuntimeError("FrequencySampler empirical cache is not prepared")
            boundary = min(set_boundary, self._cached_empirical_position_sentinel)
        else:
            if self._cached_max_boundary is None:
                raise RuntimeError("FrequencySampler prefix cache is not prepared")
            boundary = min(set_boundary, self._cached_max_boundary)
        return float(np.mean((query_positions < boundary) & (query_positions < trigger_positions)))

    def _estimate_empirical_ordering_probability(
        self,
        query: str,
        trigger: str,
    ) -> float:
        if self._cached_empirical_required_positions is None:
            raise RuntimeError("FrequencySampler empirical cache is not prepared")

        query_key = str(query).strip().lower()
        trigger_key = str(trigger).strip().lower()
        if query_key == trigger_key:
            return 0.0

        query_positions = self._get_required_positions(query_key)
        trigger_positions = self._get_required_positions(trigger_key)
        if query_positions is None or trigger_positions is None:
            return 0.0
        return float(np.mean(query_positions < trigger_positions))

    def prepare_contexts(self, contexts: Sequence[str]) -> None:
        del contexts

    def prepare_run(
        self,
        *,
        contexts: Sequence[str],
        set_boundaries: Sequence[int],
        num_reps: int,
        model_names: Sequence[str],
    ) -> None:
        del contexts
        if self._empirical_ordering:
            sampled_models = {"ordering", "set", "conjunction", "disjunction"}
        else:
            sampled_models = {"set", "conjunction", "disjunction"}
        if not any(model_name in sampled_models for model_name in model_names):
            return
        if self._empirical_ordering:
            self._ensure_global_empirical_cache(num_reps=num_reps)
            return
        if not set_boundaries:
            return
        self._ensure_global_prefix_cache(
            num_reps=num_reps,
            max_boundary=max(int(boundary) for boundary in set_boundaries),
        )

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

    def supports_direct_model(self, model_name: str) -> bool:
        return model_name in {"ordering", "set", "conjunction", "disjunction"}

    def direct_model_result(
        self,
        *,
        model_name: str,
        context: str,
        query: str,
        trigger: str,
        query_negated: int,
        set_boundary: int,
    ) -> ModelResult | None:
        del context
        query_key = str(query).strip().lower()
        trigger_key = str(trigger).strip().lower()
        cache_key = (model_name, query_key, trigger_key, int(set_boundary))
        if cache_key in self._cached_probability_lookup:
            return probability_to_model_result(
                self._cached_probability_lookup[cache_key],
                query_negated,
            )

        if model_name == "ordering":
            if self._empirical_ordering:
                negation_probability = self._estimate_empirical_ordering_probability(query, trigger)
            else:
                negation_probability = self.exact_negation_probability(
                    model_name=model_name,
                    context="",
                    query=query,
                    trigger=trigger,
                    set_boundary=set_boundary,
                )
        elif model_name == "set":
            negation_probability = self._estimate_set_probability(query, set_boundary)
        elif model_name == "conjunction":
            negation_probability = self._estimate_conjunction_probability(
                query,
                trigger,
                set_boundary,
            )
        elif model_name == "disjunction":
            set_probability = self._estimate_set_probability(query, set_boundary)
            conjunction_probability = self._estimate_conjunction_probability(
                query,
                trigger,
                set_boundary,
            )
            if self._empirical_ordering:
                ordering_probability = self._estimate_empirical_ordering_probability(query, trigger)
            else:
                ordering_probability = self.exact_negation_probability(
                    model_name="ordering",
                    context="",
                    query=query,
                    trigger=trigger,
                    set_boundary=set_boundary,
                )
                if ordering_probability is None:
                    return None
            negation_probability = float(
                np.clip(
                    set_probability + ordering_probability - conjunction_probability,
                    0.0,
                    1.0,
                )
            )
        else:
            return None

        if negation_probability is None:
            return None

        self._cached_probability_lookup[cache_key] = negation_probability
        return probability_to_model_result(negation_probability, query_negated)

    def supports_exact_model(self, model_name: str) -> bool:
        return (not self._empirical_ordering) and model_name == "ordering"

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
        total = query_count + trigger_count
        if total <= 0:
            return None
        return float(query_count) / float(total)


class QwenSampler(ContextSampler):
    """Context-specific sampler backed by precomputed Qwen continuation log-probabilities."""

    _TOP_SELECTION_CHUNK_SIZE = 1_000_000

    def __init__(
        self,
        *,
        required_tokens_by_context: Dict[str, Sequence[str]],
        log_probs_dir: str | Path,
        top_vocab_size: int = 100_000,
        empirical_ordering: bool = False,
        support_mode: str = "top_plus_required",
        seed: int | None = None,
    ) -> None:
        if top_vocab_size <= 0:
            raise ValueError("QwenSampler top_vocab_size must be > 0")
        if support_mode not in {"top_plus_required", "required_only"}:
            raise ValueError(f"Unsupported QwenSampler support_mode '{support_mode}'")

        self._rng = np.random.default_rng(seed)
        self._log_probs_dir = Path(log_probs_dir)
        self._top_vocab_size = int(top_vocab_size)
        self._empirical_ordering = bool(empirical_ordering)
        self._support_mode = str(support_mode)
        self._manifest = self._load_manifest(self._log_probs_dir / "vocab_manifest.json")
        self._total_count = int(self._manifest["total_count"])
        self._unigram_count = next(
            int(source["count"])
            for source in self._manifest["sources"]
            if str(source["name"]) == "1gram"
        )

        self._required_tokens_by_context = {
            str(context): normalize_unique_tokens(tokens)
            for context, tokens in required_tokens_by_context.items()
        }
        all_required_tokens = normalize_unique_tokens(
            [
                token
                for tokens in self._required_tokens_by_context.values()
                for token in tokens
            ]
        )
        self._token_to_global_index = self._locate_required_tokens(all_required_tokens)
        self._available_contexts = self._discover_available_contexts()

        self._context_token_log_probs: Dict[str, Dict[str, float]] = {}
        self._context_probability_lookup: Dict[str, Dict[Tuple[str, str, str, int], float]] = {}
        self._context_cache_state: Dict[str, Dict[str, object]] = {}

    @property
    def available_contexts(self) -> List[str]:
        return list(self._available_contexts)

    @staticmethod
    def _safe_stem(text: str) -> str:
        keep = []
        for char in str(text):
            if char.isalnum() or char in {"_", "-"}:
                keep.append(char)
            else:
                keep.append("_")
        return "".join(keep).strip("_")

    @staticmethod
    def _load_manifest(manifest_path: Path) -> dict:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing Qwen vocab manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        total_count = manifest.get("total_count")
        sources = manifest.get("sources")
        if not isinstance(total_count, int) or total_count <= 0:
            raise ValueError(f"Invalid total_count in manifest: {manifest_path}")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Invalid sources in manifest: {manifest_path}")
        return manifest

    def _iter_manifest_sources(self) -> List[Tuple[Path, int]]:
        sources: List[Tuple[Path, int]] = []
        for source in self._manifest["sources"]:
            source_path = Path(str(source["path"]))
            offset = int(source["offset"])
            if not source_path.exists():
                raise FileNotFoundError(f"Missing vocab source listed in manifest: {source_path}")
            sources.append((source_path, offset))
        return sources

    def _locate_required_tokens(self, tokens: Sequence[str]) -> Dict[str, int]:
        remaining = set(normalize_unique_tokens(tokens))
        if not remaining:
            return {}

        token_to_index: Dict[str, int] = {}
        for source_path, offset in self._iter_manifest_sources():
            with source_path.open("r", encoding="utf-8") as handle:
                for line_no, raw_line in enumerate(handle, start=1):
                    token = raw_line.rstrip("\n")
                    if token not in remaining:
                        continue
                    token_to_index[token] = offset + line_no - 1
                    remaining.remove(token)
                    if not remaining:
                        return token_to_index
        return token_to_index

    def _context_path(self, context: str, suffix: str) -> Path:
        return self._log_probs_dir / f"{self._safe_stem(context)}{suffix}"

    def _load_context_progress(self, context: str) -> dict | None:
        progress_path = self._context_path(context, ".progress.json")
        if not progress_path.exists():
            return None
        with progress_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _is_context_complete(self, context: str) -> bool:
        output_path = self._context_path(context, ".log_probs.npy")
        progress = self._load_context_progress(context)
        if progress is None or not output_path.exists():
            return False

        sources = progress.get("sources", {})
        if not isinstance(sources, dict) or not sources:
            return False
        if not all(bool(source_info.get("done")) for source_info in sources.values()):
            return False

        bigram_progress = sources.get("2gram")
        if not isinstance(bigram_progress, dict):
            return False

        selection_mode = bigram_progress.get("selection_mode")
        if selection_mode in {
            "sparse_top_support",
            "context_balanced_support",
            "context_balanced_support_global_union",
        }:
            target_vocab_size = bigram_progress.get("target_vocab_size")
            required_target = 1 if self._support_mode == "required_only" else self._top_vocab_size
            if target_vocab_size is None or int(target_vocab_size) < required_target:
                return False
            required_missing = bigram_progress.get("required_missing", [])
            if required_missing:
                return False

        return True

    def _discover_available_contexts(self) -> List[str]:
        available = []
        for context in self._required_tokens_by_context:
            if self._is_context_complete(context):
                available.append(context)
        return sorted(available)

    def _load_context_log_probs(self, context: str) -> np.ndarray:
        output_path = self._context_path(context, ".log_probs.npy")
        if not output_path.exists():
            raise FileNotFoundError(f"Missing Qwen log-prob array for context '{context}': {output_path}")
        log_probs = np.load(output_path, mmap_mode="r")
        if len(log_probs) != self._total_count:
            raise ValueError(
                f"Unexpected Qwen vocab size for context '{context}': "
                f"expected {self._total_count}, got {len(log_probs)}"
            )
        return log_probs

    def _ensure_context_token_log_probs(self, context: str) -> None:
        key = str(context)
        if key in self._context_token_log_probs:
            return
        if key not in self._available_contexts:
            self._context_token_log_probs[key] = {}
            return

        required_tokens = self._required_tokens_by_context.get(key, [])
        log_probs = self._load_context_log_probs(key)
        token_log_probs: Dict[str, float] = {}
        for token in required_tokens:
            token_idx = self._token_to_global_index.get(token)
            if token_idx is None:
                continue
            value = float(log_probs[token_idx])
            if np.isfinite(value):
                token_log_probs[token] = value
        self._context_token_log_probs[key] = token_log_probs

    @classmethod
    def _top_finite_indices(
        cls,
        log_probs: np.ndarray,
        limit: int,
    ) -> np.ndarray:
        if limit <= 0:
            return np.empty(0, dtype=np.int64)

        best_indices = np.empty(0, dtype=np.int64)
        best_values = np.empty(0, dtype=np.float32)
        total_count = len(log_probs)

        for start in range(0, total_count, cls._TOP_SELECTION_CHUNK_SIZE):
            stop = min(start + cls._TOP_SELECTION_CHUNK_SIZE, total_count)
            chunk = np.asarray(log_probs[start:stop], dtype=np.float32)
            finite_mask = np.isfinite(chunk)
            if not finite_mask.any():
                continue

            local_indices = np.flatnonzero(finite_mask)
            local_values = chunk[local_indices]
            local_global_indices = local_indices.astype(np.int64) + start

            if len(local_values) > limit:
                selected = np.argpartition(local_values, -limit)[-limit:]
                local_values = local_values[selected]
                local_global_indices = local_global_indices[selected]

            if len(best_values) == 0:
                best_values = local_values
                best_indices = local_global_indices
            else:
                combined_values = np.concatenate([best_values, local_values])
                combined_indices = np.concatenate([best_indices, local_global_indices])
                if len(combined_values) > limit:
                    selected = np.argpartition(combined_values, -limit)[-limit:]
                    best_values = combined_values[selected]
                    best_indices = combined_indices[selected]
                else:
                    best_values = combined_values
                    best_indices = combined_indices

        if len(best_values) == 0:
            return np.empty(0, dtype=np.int64)

        ordering = np.lexsort((best_indices, -best_values))
        return best_indices[ordering]

    def _build_context_support(
        self,
        context: str,
        log_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        required_tokens = self._required_tokens_by_context.get(context, [])
        required_token_to_index = {
            token: idx for idx, token in enumerate(required_tokens)
        }
        required_indices = []
        for token in required_tokens:
            token_idx = self._token_to_global_index.get(token)
            if token_idx is None:
                continue
            if token_idx >= len(log_probs):
                continue
            if np.isfinite(log_probs[token_idx]):
                required_indices.append(token_idx)

        if self._support_mode == "required_only":
            merged_indices = np.asarray(required_indices, dtype=np.int64)
        else:
            support_indices = self._top_finite_indices(log_probs, self._top_vocab_size)
            merged_indices = np.unique(
                np.concatenate(
                    [
                        support_indices,
                        np.asarray(required_indices, dtype=np.int64),
                    ]
                )
            )
        if len(merged_indices) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                required_token_to_index,
            )

        support_log_probs = np.asarray(log_probs[merged_indices], dtype=np.float64)
        finite_mask = np.isfinite(support_log_probs)
        merged_indices = merged_indices[finite_mask]
        support_log_probs = support_log_probs[finite_mask]
        if len(merged_indices) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                required_token_to_index,
            )

        ordering = np.lexsort((merged_indices, -support_log_probs))
        merged_indices = merged_indices[ordering]
        support_log_probs = support_log_probs[ordering]

        shifted = support_log_probs - support_log_probs.max()
        support_probs = np.exp(shifted)
        support_probs = support_probs / support_probs.sum()
        return merged_indices, support_probs.astype(np.float64), required_token_to_index

    def _ensure_context_prefix_cache(
        self,
        context: str,
        *,
        num_reps: int,
        max_boundary: int,
    ) -> None:
        if max_boundary <= 0:
            raise ValueError("QwenSampler max_boundary must be > 0")

        key = str(context)
        state = self._context_cache_state.get(key)
        if (
            state is not None
            and state.get("num_reps") == num_reps
            and state.get("max_boundary") == max_boundary
        ):
            return

        if key not in self._available_contexts:
            self._context_cache_state[key] = {
                "num_reps": num_reps,
                "max_boundary": max_boundary,
                "cache_boundary": 0,
                "position_sentinel": 0,
                "required_positions": None,
                "required_token_to_index": {
                    token: idx
                    for idx, token in enumerate(self._required_tokens_by_context.get(key, []))
                },
            }
            self._context_probability_lookup[key] = {}
            return

        log_probs = self._load_context_log_probs(key)
        support_indices, support_probs, required_token_to_index = self._build_context_support(
            key,
            log_probs,
        )
        cache_boundary = min(max_boundary, len(support_indices))
        sentinel = cache_boundary
        dtype = np.int16 if sentinel <= np.iinfo(np.int16).max else np.int32
        required_positions = np.full(
            (num_reps, len(required_token_to_index)),
            sentinel,
            dtype=dtype,
        )

        global_index_to_required_index = {
            self._token_to_global_index[token]: required_idx
            for token, required_idx in required_token_to_index.items()
            if token in self._token_to_global_index
        }
        support_required_lookup = np.full(len(support_indices), -1, dtype=np.int32)
        for support_position, global_index in enumerate(support_indices):
            required_idx = global_index_to_required_index.get(int(global_index))
            if required_idx is not None:
                support_required_lookup[support_position] = required_idx

        for rep_idx in range(num_reps):
            sampled_support_positions = self._rng.choice(
                len(support_indices),
                size=cache_boundary,
                replace=False,
                p=support_probs,
            )
            for sampled_position, support_position in enumerate(sampled_support_positions):
                required_idx = support_required_lookup[int(support_position)]
                if required_idx >= 0:
                    required_positions[rep_idx, required_idx] = sampled_position

        self._context_cache_state[key] = {
            "num_reps": num_reps,
            "max_boundary": max_boundary,
            "cache_boundary": cache_boundary,
            "position_sentinel": sentinel,
            "required_positions": required_positions,
            "required_token_to_index": required_token_to_index,
        }
        self._context_probability_lookup[key] = {}

    def _ensure_context_empirical_cache(
        self,
        context: str,
        *,
        num_reps: int,
    ) -> None:
        key = str(context)
        state = self._context_cache_state.get(key)
        if (
            state is not None
            and state.get("cache_mode") == "empirical"
            and state.get("num_reps") == num_reps
        ):
            return

        if key not in self._available_contexts:
            self._context_cache_state[key] = {
                "cache_mode": "empirical",
                "num_reps": num_reps,
                "cache_boundary": 0,
                "position_sentinel": 0,
                "required_positions": None,
                "required_token_to_index": {
                    token: idx
                    for idx, token in enumerate(self._required_tokens_by_context.get(key, []))
                },
            }
            self._context_probability_lookup[key] = {}
            return

        log_probs = self._load_context_log_probs(key)
        support_indices, support_probs, required_token_to_index = self._build_context_support(
            key,
            log_probs,
        )
        cache_boundary = len(support_indices)
        sentinel = cache_boundary
        dtype = np.int16 if sentinel <= np.iinfo(np.int16).max else np.int32
        required_positions = np.full(
            (num_reps, len(required_token_to_index)),
            sentinel,
            dtype=dtype,
        )

        global_index_to_required_index = {
            self._token_to_global_index[token]: required_idx
            for token, required_idx in required_token_to_index.items()
            if token in self._token_to_global_index
        }
        support_required_lookup = np.full(len(support_indices), -1, dtype=np.int32)
        for support_position, global_index in enumerate(support_indices):
            required_idx = global_index_to_required_index.get(int(global_index))
            if required_idx is not None:
                support_required_lookup[support_position] = required_idx

        for rep_idx in range(num_reps):
            sampled_support_positions = _sample_weighted_indices(self._rng, support_probs)
            for sampled_position, support_position in enumerate(sampled_support_positions):
                required_idx = support_required_lookup[int(support_position)]
                if required_idx >= 0:
                    required_positions[rep_idx, required_idx] = sampled_position

        self._context_cache_state[key] = {
            "cache_mode": "empirical",
            "num_reps": num_reps,
            "cache_boundary": cache_boundary,
            "position_sentinel": sentinel,
            "required_positions": required_positions,
            "required_token_to_index": required_token_to_index,
        }
        self._context_probability_lookup[key] = {}

    def _get_required_positions(self, context: str, token: str) -> np.ndarray | None:
        key = str(context)
        state = self._context_cache_state.get(key)
        if state is None:
            return None
        required_positions = state.get("required_positions")
        if required_positions is None:
            return None
        required_token_to_index = state.get("required_token_to_index", {})
        token_idx = required_token_to_index.get(str(token).strip().lower())
        if token_idx is None:
            return None
        return required_positions[:, token_idx]

    def _estimate_set_probability(self, context: str, query: str, set_boundary: int) -> float:
        state = self._context_cache_state.get(str(context))
        if state is None:
            raise RuntimeError(f"QwenSampler cache not prepared for context '{context}'")

        query_positions = self._get_required_positions(context, query)
        if query_positions is None:
            return 0.0

        boundary = min(int(set_boundary), int(state["cache_boundary"]))
        return float(np.mean(query_positions < boundary))

    def _estimate_conjunction_probability(
        self,
        context: str,
        query: str,
        trigger: str,
        set_boundary: int,
    ) -> float:
        state = self._context_cache_state.get(str(context))
        if state is None:
            raise RuntimeError(f"QwenSampler cache not prepared for context '{context}'")

        query_positions = self._get_required_positions(context, query)
        if query_positions is None:
            return 0.0

        trigger_positions = self._get_required_positions(context, trigger)
        if trigger_positions is None:
            trigger_positions = np.full_like(query_positions, int(state["position_sentinel"]))

        boundary = min(int(set_boundary), int(state["cache_boundary"]))
        return float(np.mean((query_positions < boundary) & (query_positions < trigger_positions)))

    def _estimate_empirical_ordering_probability(
        self,
        context: str,
        query: str,
        trigger: str,
    ) -> float:
        state = self._context_cache_state.get(str(context))
        if state is None:
            raise RuntimeError(f"QwenSampler cache not prepared for context '{context}'")

        query_key = str(query).strip().lower()
        trigger_key = str(trigger).strip().lower()
        if query_key == trigger_key:
            return 0.0

        query_positions = self._get_required_positions(context, query_key)
        trigger_positions = self._get_required_positions(context, trigger_key)
        if query_positions is None or trigger_positions is None:
            return 0.0
        return float(np.mean(query_positions < trigger_positions))

    def prepare_run(
        self,
        *,
        contexts: Sequence[str],
        set_boundaries: Sequence[int],
        num_reps: int,
        model_names: Sequence[str],
    ) -> None:
        if self._empirical_ordering:
            sampled_models = {"ordering", "set", "conjunction", "disjunction"}
            needs_samples = any(model_name in sampled_models for model_name in model_names)
            max_boundary = 0
        else:
            sampled_models = {"set", "conjunction", "disjunction"}
            needs_samples = any(model_name in sampled_models for model_name in model_names)
            max_boundary = max((int(boundary) for boundary in set_boundaries), default=0)

        for context in contexts:
            key = str(context)
            self._ensure_context_token_log_probs(key)
            if self._empirical_ordering and needs_samples:
                self._ensure_context_empirical_cache(
                    key,
                    num_reps=num_reps,
                )
            elif needs_samples and max_boundary > 0:
                self._ensure_context_prefix_cache(
                    key,
                    num_reps=num_reps,
                    max_boundary=max_boundary,
                )

    def sample_contexts(
        self,
        contexts: Sequence[str],
        set_boundary: int,
        num_reps: int,
    ) -> ContextSamples:
        del contexts, set_boundary, num_reps
        raise NotImplementedError(
            "QwenSampler scores current models directly and does not expose raw sampled orderings"
        )

    def supports_token(self, context: str, token: str) -> bool:
        key = str(context)
        token_key = str(token).strip().lower()
        self._ensure_context_token_log_probs(key)
        return token_key in self._context_token_log_probs.get(key, {})

    def supports_direct_model(self, model_name: str) -> bool:
        return model_name in {"ordering", "set", "conjunction", "disjunction"}

    def direct_model_result(
        self,
        *,
        model_name: str,
        context: str,
        query: str,
        trigger: str,
        query_negated: int,
        set_boundary: int,
    ) -> ModelResult | None:
        context_key = str(context)
        query_key = str(query).strip().lower()
        trigger_key = str(trigger).strip().lower()
        cache_key = (model_name, query_key, trigger_key, int(set_boundary))
        context_lookup = self._context_probability_lookup.setdefault(context_key, {})
        if cache_key in context_lookup:
            return probability_to_model_result(context_lookup[cache_key], query_negated)

        if model_name == "ordering":
            if self._empirical_ordering:
                negation_probability = self._estimate_empirical_ordering_probability(
                    context_key,
                    query_key,
                    trigger_key,
                )
            else:
                negation_probability = self.exact_negation_probability(
                    model_name="ordering",
                    context=context_key,
                    query=query_key,
                    trigger=trigger_key,
                    set_boundary=set_boundary,
                )
        elif model_name == "set":
            negation_probability = self._estimate_set_probability(context_key, query_key, set_boundary)
        elif model_name == "conjunction":
            negation_probability = self._estimate_conjunction_probability(
                context_key,
                query_key,
                trigger_key,
                set_boundary,
            )
        elif model_name == "disjunction":
            set_probability = self._estimate_set_probability(context_key, query_key, set_boundary)
            conjunction_probability = self._estimate_conjunction_probability(
                context_key,
                query_key,
                trigger_key,
                set_boundary,
            )
            if self._empirical_ordering:
                ordering_probability = self._estimate_empirical_ordering_probability(
                    context_key,
                    query_key,
                    trigger_key,
                )
            else:
                ordering_probability = self.exact_negation_probability(
                    model_name="ordering",
                    context=context_key,
                    query=query_key,
                    trigger=trigger_key,
                    set_boundary=set_boundary,
                )
                if ordering_probability is None:
                    return None
            negation_probability = float(
                np.clip(
                    set_probability + ordering_probability - conjunction_probability,
                    0.0,
                    1.0,
                )
            )
        else:
            return None

        if negation_probability is None:
            return None

        context_lookup[cache_key] = negation_probability
        return probability_to_model_result(negation_probability, query_negated)

    def supports_exact_model(self, model_name: str) -> bool:
        return (not self._empirical_ordering) and model_name == "ordering"

    def exact_negation_probability(
        self,
        *,
        model_name: str,
        context: str,
        query: str,
        trigger: str,
        set_boundary: int,
    ) -> float | None:
        del set_boundary
        if model_name != "ordering":
            return None

        query_key = str(query).strip().lower()
        trigger_key = str(trigger).strip().lower()
        if query_key == trigger_key:
            return 0.0

        self._ensure_context_token_log_probs(context)
        token_log_probs = self._context_token_log_probs.get(str(context), {})
        query_log_prob = token_log_probs.get(query_key)
        trigger_log_prob = token_log_probs.get(trigger_key)
        if query_log_prob is None or trigger_log_prob is None:
            return None

        log_ratio = float(trigger_log_prob) - float(query_log_prob)
        return float(1.0 / (1.0 + np.exp(log_ratio)))


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
