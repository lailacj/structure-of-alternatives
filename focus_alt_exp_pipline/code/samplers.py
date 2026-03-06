"""Dataset-specific sample generation.

This module owns one responsibility:
given contexts + set boundary, generate sampled orderings/sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


# class _BaseBertSampler(ContextSampler):
#     """Shared utilities for BERT-based samplers."""

#     def __init__(
#         self,
#         *,
#         model_name: str,
#         seed: int | None = None,
#         device: str | None = None,
#     ) -> None:
#         import torch
#         from transformers import AutoModelForMaskedLM, AutoTokenizer

#         self._torch = torch
#         self._rng = np.random.default_rng(seed)
#         self._device = (
#             device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
#         )
#         self._tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self._model = AutoModelForMaskedLM.from_pretrained(model_name).to(self._device).eval()

#     def _filter_non_special_distribution(
#         self,
#         distribution: List[Tuple[str, float]],
#     ) -> List[Tuple[str, float]]:
#         special_tokens = set(self._tokenizer.all_special_tokens)
#         filtered = [
#             (token, prob)
#             for token, prob in distribution
#             if not token.startswith("[") and not token.startswith("<") and token not in special_tokens
#         ]
#         if not filtered:
#             raise ValueError("Filtering removed every token from the BERT vocabulary distribution")

#         total_prob = sum(prob for _, prob in filtered)
#         if total_prob <= 0:
#             raise ValueError("Filtered BERT vocabulary distribution has non-positive total probability")

#         return [(token, prob / total_prob) for token, prob in filtered]

#     def _sample_ordering(self, distribution: List[Tuple[str, float]]) -> List[str]:
#         tokens = [tok for tok, _ in distribution]
#         probs = np.array([p for _, p in distribution], dtype=float)
#         probs = probs / probs.sum()
#         idx = self._rng.choice(len(tokens), size=len(tokens), replace=False, p=probs)
#         return [tokens[i] for i in idx]

# class BertSampler(_BaseBertSampler):
#     """Sampler for BERT masked-token distributions conditioned on context."""

#     def __init__(
#         self,
#         prompts_df: pd.DataFrame,
#         *,
#         context_col: str = "story",
#         prompt_col: str = "prompt",
#         model_name: str = "google-bert/bert-large-uncased-whole-word-masking",
#         seed: int | None = None,
#         device: str | None = None,
#     ) -> None:
#         required = {context_col, prompt_col}
#         missing = required.difference(prompts_df.columns)
#         if missing:
#             raise ValueError(f"Missing columns in prompts_df: {sorted(missing)}")

#         super().__init__(model_name=model_name, seed=seed, device=device)
#         self._prompts = {
#             str(row[context_col]): str(row[prompt_col])
#             for _, row in prompts_df[[context_col, prompt_col]].drop_duplicates().iterrows()
#         }
#         self._distribution_cache: Dict[str, List[Tuple[str, float]]] = {}
#         self._token_set_cache: Dict[str, set[str]] = {}

#     def _get_distribution(self, context: str) -> List[Tuple[str, float]]:
#         key = str(context)
#         if key in self._distribution_cache:
#             return self._distribution_cache[key]
#         prompt = self._prompts.get(key)
#         if prompt is None:
#             raise KeyError(f"No prompt found for context '{key}'")

#         inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
#         mask_positions = (inputs.input_ids == self._tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
#         if len(mask_positions) != 1:
#             raise ValueError(
#                 f"Prompt for context '{key}' must contain exactly one [MASK], got {len(mask_positions)}"
#             )

#         with self._torch.no_grad():
#             logits = self._model(**inputs).logits
#         mask_logits = logits[0, mask_positions, :].squeeze(0)
#         probs = self._torch.softmax(mask_logits, dim=-1).detach().cpu().numpy()
#         tokens = self._tokenizer.convert_ids_to_tokens(range(mask_logits.shape[-1]))
#         distribution = self._filter_non_special_distribution(list(zip(tokens, probs.tolist())))

#         self._distribution_cache[key] = distribution
#         self._token_set_cache[key] = {token for token, _ in distribution}
#         return distribution

#     def prepare_contexts(self, contexts: Sequence[str]) -> None:
#         for context in contexts:
#             self._get_distribution(str(context))

#     def sample_contexts(
#         self,
#         contexts: Sequence[str],
#         set_boundary: int,
#         num_reps: int,
#     ) -> ContextSamples:
#         context_samples: ContextSamples = {}
#         for context in contexts:
#             key = str(context)
#             distribution = self._get_distribution(key)
#             sims: List[Sample] = []
#             for _ in range(num_reps):
#                 ordering = self._sample_ordering(distribution)
#                 sims.append((ordering, ordering[:set_boundary]))
#             context_samples[key] = sims
#         return context_samples

#     def supports_token(self, context: str, token: str) -> bool:
#         key = str(context)
#         if key not in self._token_set_cache:
#             self._get_distribution(key)
#         return token in self._token_set_cache.get(key, set())

# class StaticBERTSampler(_BaseBertSampler):
#     """Sampler for a context-free BERT vocabulary distribution.

#     The distribution is computed once from a prompt containing only BERT's [MASK]
#     token, then reused for every experimental context.
#     """

#     def __init__(
#         self,
#         *,
#         model_name: str = "google-bert/bert-large-uncased-whole-word-masking",
#         seed: int | None = None,
#         device: str | None = None,
#     ) -> None:
#         super().__init__(model_name=model_name, seed=seed, device=device)
#         self._distribution: List[Tuple[str, float]] | None = None
#         self._token_set: set[str] = set()

#     def _get_distribution(self) -> List[Tuple[str, float]]:
#         if self._distribution is not None:
#             return self._distribution

#         inputs = self._tokenizer(self._tokenizer.mask_token, return_tensors="pt").to(self._device)
#         mask_positions = (inputs.input_ids == self._tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
#         if len(mask_positions) != 1:
#             raise ValueError(
#                 f"Static BERT prompt must contain exactly one [MASK], got {len(mask_positions)}"
#             )

#         with self._torch.no_grad():
#             logits = self._model(**inputs).logits
#         mask_logits = logits[0, mask_positions, :].squeeze(0)
#         probs = self._torch.softmax(mask_logits, dim=-1).detach().cpu().numpy()
#         tokens = self._tokenizer.convert_ids_to_tokens(range(mask_logits.shape[-1]))
#         distribution = self._filter_non_special_distribution(list(zip(tokens, probs.tolist())))

#         self._distribution = distribution
#         self._token_set = {token for token, _ in distribution}
#         return distribution

#     def prepare_contexts(self, contexts: Sequence[str]) -> None:
#         del contexts
#         self._get_distribution()

#     def sample_contexts(
#         self,
#         contexts: Sequence[str],
#         set_boundary: int,
#         num_reps: int,
#     ) -> ContextSamples:
#         distribution = self._get_distribution()
#         context_samples: ContextSamples = {}
#         for context in contexts:
#             key = str(context)
#             sims: List[Sample] = []
#             for _ in range(num_reps):
#                 ordering = self._sample_ordering(distribution)
#                 sims.append((ordering, ordering[:set_boundary]))
#             context_samples[key] = sims
#         return context_samples

#     def supports_token(self, context: str, token: str) -> bool:
#         del context
#         if not self._token_set:
#             self._get_distribution()
#         return token in self._token_set
