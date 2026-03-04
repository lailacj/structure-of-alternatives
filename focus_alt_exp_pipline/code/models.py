"""Model-specific scoring functions and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

Sample = Tuple[Sequence[str], Sequence[str]]
ModelResult = Tuple[float, float, float]
ModelFn = Callable[[Sequence[Sample], str, str, int], ModelResult]


def _to_log_likelihood(negation_probability: float, query_negated: int) -> ModelResult:
    if query_negated == 1:
        prob_query_observed = negation_probability
    else:
        prob_query_observed = 1 - negation_probability
    
    if prob_query_observed > 0:
        log_likelihood = np.log(prob_query_observed)
    else:
        log_likelihood = np.log(1e-10)

    return log_likelihood, negation_probability, prob_query_observed


def ordering_model(
    samples: Sequence[Sample],
    query: str,
    trigger: str,
    query_negated: int,
) -> ModelResult:

    count_query_above_trigger = 0
    for sampled_ordering, _ in samples:
        try:
            query_index = sampled_ordering.index(query)
            trigger_index = sampled_ordering.index(trigger)
        except ValueError:
            continue
        if query_index < trigger_index:
            count_query_above_trigger += 1

    negation_probability = float(count_query_above_trigger) / float(len(samples))
    return _to_log_likelihood(negation_probability, query_negated)


def set_model(
    samples: Sequence[Sample],
    query: str,
    trigger: str,
    query_negated: int,
) -> ModelResult:

    del trigger
    count_query_in_set = 0
    for _, sampled_set in samples:
        if query in sampled_set:
            count_query_in_set += 1

    negation_probability = float(count_query_in_set) / float(len(samples))
    return _to_log_likelihood(negation_probability, query_negated)


def conjunction_model(
    samples: Sequence[Sample],
    query: str,
    trigger: str,
    query_negated: int,
) -> ModelResult:

    count_conjunction = 0
    for sampled_ordering, sampled_set in samples:
        if query not in sampled_set:
            continue
        try:
            query_index = sampled_ordering.index(query)
            trigger_index = sampled_ordering.index(trigger)
        except ValueError:
            continue
        if query_index < trigger_index:
            count_conjunction += 1

    negation_probability = float(count_conjunction) / float(len(samples))
    return _to_log_likelihood(negation_probability, query_negated)


def disjunction_model(
    samples: Sequence[Sample],
    query: str,
    trigger: str,
    query_negated: int,
) -> ModelResult:

    count_disjunction = 0
    for sampled_ordering, sampled_set in samples:
        try:
            query_index = sampled_ordering.index(query)
            trigger_index = sampled_ordering.index(trigger)
        except ValueError:
            continue
        if ((query in sampled_set) or (query_index < trigger_index)):
            count_disjunction += 1

    negation_probability = float(count_disjunction) / float(len(samples))
    return _to_log_likelihood(negation_probability, query_negated)


def always_negate_model(
    samples: Sequence[Sample],
    query: str,
    trigger: str,
    query_negated: int,
) -> ModelResult:

    del samples, query, trigger
    return _to_log_likelihood(1.0, query_negated)


def never_negate_model(
    samples: Sequence[Sample],
    query: str,
    trigger: str,
    query_negated: int,
) -> ModelResult:

    del samples, query, trigger
    return _to_log_likelihood(0.0, query_negated)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fn: ModelFn
    requires_trigger: bool = True


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "ordering": ModelSpec("ordering", ordering_model, requires_trigger=True),
    "set": ModelSpec("set", set_model, requires_trigger=False),
    "conjunction": ModelSpec("conjunction", conjunction_model, requires_trigger=True),
    "disjunction": ModelSpec("disjunction", disjunction_model, requires_trigger=True),
    "always_negate": ModelSpec("always_negate", always_negate_model, requires_trigger=False),
    "never_negate": ModelSpec("never_negate", never_negate_model, requires_trigger=False),
}


def get_models(
    *,
    model_names: Iterable[str] | None = None,
    include_baselines: bool = False,
) -> List[ModelSpec]:

    if model_names is None:
        base = ["ordering", "set", "conjunction", "disjunction"]
        if include_baselines:
            base.extend(["always_negate", "never_negate"])
        model_names = base

    selected: List[ModelSpec] = []
    for name in model_names:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{name}'")
        selected.append(MODEL_REGISTRY[name])
    return selected
