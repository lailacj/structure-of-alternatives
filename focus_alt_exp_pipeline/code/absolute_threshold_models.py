"""Absolute-expectedness-threshold alternative-structure probabilities.

This module implements the paper's intended Set model independently of the
legacy top-K sampler.  For a context ``c``, let ``query_logprob`` and
``trigger_logprob`` be summed Qwen continuation log probabilities in the same
neutral, no-frame completion context.  Latent utilities are

    u_x = log P_Qwen(x | c) + epsilon_x,

where the epsilon terms are independent Gumbel random variables with a shared
scale.  A candidate is in the alternative set when its latent utility exceeds
an absolute threshold.  The four events are:

    set:         u_query >= threshold
    ordering:    u_query > u_trigger
    conjunction: set and ordering
    disjunction: set or ordering

The functions below evaluate the corresponding probabilities analytically.
They do not sample or construct a fixed-size candidate set, so different
contexts may imply differently sized alternative sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Probability: TypeAlias = float | NDArray[np.float64]


@dataclass(frozen=True)
class StructureProbabilities:
    """Predicted query-exclusion probabilities for the four structures."""

    set: Probability
    ordering: Probability
    conjunction: Probability
    disjunction: Probability


def _validate_scale(scale: float) -> float:
    resolved = float(scale)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError("scale must be a finite number greater than zero")
    return resolved


def _as_finite_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _restore_scalar(array: NDArray[np.float64]) -> Probability:
    if array.ndim == 0:
        return float(array)
    return array


def _gumbel_survival_from_log_rate(log_rate: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``1 - exp(-exp(log_rate))`` without numerical overflow."""

    # exp(50) is already large enough for the survival probability to round to
    # one.  -745 is close to the smallest representable float exponent.
    safe_log_rate = np.clip(log_rate, -745.0, 50.0)
    rate = np.exp(safe_log_rate)
    return np.clip(-np.expm1(-rate), 0.0, 1.0)


def _stable_sigmoid(value: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(value, dtype=np.float64)
    nonnegative = value >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    result[~nonnegative] = exp_value / (1.0 + exp_value)
    return result


def set_inclusion_probability(
    query_logprob: ArrayLike,
    *,
    threshold: float,
    scale: float = 1.0,
) -> Probability:
    """Probability that the query exceeds the absolute expectedness threshold."""

    resolved_scale = _validate_scale(scale)
    query = _as_finite_array(query_logprob, name="query_logprob")
    resolved_threshold = float(threshold)
    if not np.isfinite(resolved_threshold):
        raise ValueError("threshold must be finite")

    log_rate = (query - resolved_threshold) / resolved_scale
    probability = _gumbel_survival_from_log_rate(log_rate)
    return _restore_scalar(probability)


def ordering_probability(
    query_logprob: ArrayLike,
    trigger_logprob: ArrayLike,
    *,
    scale: float = 1.0,
) -> Probability:
    """Probability that the query's latent utility exceeds the trigger's."""

    resolved_scale = _validate_scale(scale)
    query = _as_finite_array(query_logprob, name="query_logprob")
    trigger = _as_finite_array(trigger_logprob, name="trigger_logprob")
    query, trigger = np.broadcast_arrays(query, trigger)

    probability = _stable_sigmoid((query - trigger) / resolved_scale)
    return _restore_scalar(probability)


def structure_probabilities(
    query_logprob: ArrayLike,
    trigger_logprob: ArrayLike,
    *,
    threshold: float,
    scale: float = 1.0,
) -> StructureProbabilities:
    """Return Set, Ordering, Conjunction, and Disjunction probabilities.

    Inputs may be scalars or broadcast-compatible NumPy-like arrays.  The
    conjunction is evaluated as one joint event under shared latent utilities;
    it is not the product of the marginal Set and Ordering probabilities.
    """

    resolved_scale = _validate_scale(scale)
    query = _as_finite_array(query_logprob, name="query_logprob")
    trigger = _as_finite_array(trigger_logprob, name="trigger_logprob")
    query, trigger = np.broadcast_arrays(query, trigger)

    resolved_threshold = float(threshold)
    if not np.isfinite(resolved_threshold):
        raise ValueError("threshold must be finite")

    query_log_rate = (query - resolved_threshold) / resolved_scale
    trigger_log_rate = (trigger - resolved_threshold) / resolved_scale

    set_probability = _gumbel_survival_from_log_rate(query_log_rate)
    order_probability = _stable_sigmoid((query - trigger) / resolved_scale)

    # The conjunction event is that the query is the higher-utility member of
    # the pair and that this winning utility exceeds the threshold.
    either_above_probability = _gumbel_survival_from_log_rate(
        np.logaddexp(query_log_rate, trigger_log_rate)
    )
    conjunction_probability = np.clip(
        order_probability * either_above_probability,
        0.0,
        1.0,
    )
    disjunction_probability = np.clip(
        set_probability + order_probability - conjunction_probability,
        0.0,
        1.0,
    )

    return StructureProbabilities(
        set=_restore_scalar(set_probability),
        ordering=_restore_scalar(order_probability),
        conjunction=_restore_scalar(conjunction_probability),
        disjunction=_restore_scalar(disjunction_probability),
    )
