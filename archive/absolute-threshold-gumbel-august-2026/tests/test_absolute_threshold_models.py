"""Tests for the absolute-threshold alternative-structure model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from absolute_threshold_models import (  # noqa: E402
    ordering_probability,
    set_inclusion_probability,
    structure_probabilities,
)


class AbsoluteThresholdModelTests(unittest.TestCase):
    def test_ordering_is_pairwise_softmax(self) -> None:
        query = -2.0
        trigger = -3.0

        observed = ordering_probability(query, trigger)
        expected = np.exp(query) / (np.exp(query) + np.exp(trigger))

        self.assertAlmostEqual(observed, expected)

    def test_equal_scores_give_even_ordering_probability(self) -> None:
        self.assertAlmostEqual(ordering_probability(-4.0, -4.0), 0.5)

    def test_set_probability_increases_with_absolute_expectedness(self) -> None:
        probabilities = set_inclusion_probability(
            np.array([-8.0, -4.0, -1.0]),
            threshold=-4.0,
        )

        self.assertTrue(np.all(np.diff(probabilities) > 0))

    def test_same_threshold_allows_contexts_to_have_different_expected_set_sizes(self) -> None:
        high_expectedness_context = np.array([-1.0, -2.0, -3.0, -4.0])
        low_expectedness_context = np.array([-6.0, -7.0, -8.0, -9.0])

        high_size = np.sum(
            set_inclusion_probability(high_expectedness_context, threshold=-4.0)
        )
        low_size = np.sum(
            set_inclusion_probability(low_expectedness_context, threshold=-4.0)
        )

        self.assertGreater(high_size, low_size)

    def test_hybrid_probabilities_obey_event_bounds_and_inclusion_exclusion(self) -> None:
        result = structure_probabilities(
            query_logprob=np.array([-5.0, -2.0, -1.0]),
            trigger_logprob=np.array([-3.0, -3.0, -4.0]),
            threshold=-3.0,
        )

        self.assertTrue(np.all(result.conjunction <= result.set))
        self.assertTrue(np.all(result.conjunction <= result.ordering))
        self.assertTrue(np.all(result.disjunction >= result.set))
        self.assertTrue(np.all(result.disjunction >= result.ordering))
        np.testing.assert_allclose(
            result.disjunction,
            result.set + result.ordering - result.conjunction,
        )

    def test_closed_form_matches_latent_utility_simulation(self) -> None:
        query = -1.2
        trigger = -1.8
        threshold = -1.5
        expected = structure_probabilities(query, trigger, threshold=threshold)

        rng = np.random.default_rng(11)
        num_samples = 200_000
        query_utility = query + rng.gumbel(size=num_samples)
        trigger_utility = trigger + rng.gumbel(size=num_samples)
        set_event = query_utility >= threshold
        ordering_event = query_utility > trigger_utility

        self.assertAlmostEqual(expected.set, float(np.mean(set_event)), delta=0.006)
        self.assertAlmostEqual(
            expected.ordering,
            float(np.mean(ordering_event)),
            delta=0.006,
        )
        self.assertAlmostEqual(
            expected.conjunction,
            float(np.mean(set_event & ordering_event)),
            delta=0.006,
        )
        self.assertAlmostEqual(
            expected.disjunction,
            float(np.mean(set_event | ordering_event)),
            delta=0.006,
        )

    def test_extreme_logprobabilities_remain_finite(self) -> None:
        result = structure_probabilities(
            query_logprob=np.array([-10_000.0, 10_000.0]),
            trigger_logprob=np.array([10_000.0, -10_000.0]),
            threshold=0.0,
        )

        for probability in [
            result.set,
            result.ordering,
            result.conjunction,
            result.disjunction,
        ]:
            self.assertTrue(np.all(np.isfinite(probability)))
            self.assertTrue(np.all(probability >= 0.0))
            self.assertTrue(np.all(probability <= 1.0))

    def test_array_inputs_broadcast(self) -> None:
        result = structure_probabilities(
            query_logprob=np.array([-3.0, -2.0, -1.0]),
            trigger_logprob=-2.5,
            threshold=-2.0,
        )

        self.assertEqual(result.set.shape, (3,))
        self.assertEqual(result.ordering.shape, (3,))

    def test_invalid_scale_is_rejected(self) -> None:
        for scale in [0.0, -1.0, np.inf, np.nan]:
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    structure_probabilities(-2.0, -3.0, threshold=-4.0, scale=scale)

    def test_nonfinite_scores_and_threshold_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            structure_probabilities(np.nan, -3.0, threshold=-4.0)
        with self.assertRaises(ValueError):
            structure_probabilities(-2.0, np.inf, threshold=-4.0)
        with self.assertRaises(ValueError):
            structure_probabilities(-2.0, -3.0, threshold=np.nan)


if __name__ == "__main__":
    unittest.main()
