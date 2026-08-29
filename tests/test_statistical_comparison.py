"""Tests for paired stratified bootstrap uncertainty."""

from __future__ import annotations

import unittest

import numpy as np

from src.statistical_comparison import bootstrap_model_comparison


class StatisticalComparisonTests(unittest.TestCase):
    def test_intervals_are_reproducible_and_paired(self) -> None:
        y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        predictions = {
            "strong": np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
            "weak": np.asarray([0, 1, 0, 0, 1, 0, 1, 0]),
        }
        first = bootstrap_model_comparison(
            y_true,
            predictions,
            n_resamples=100,
            random_state=7,
        )
        second = bootstrap_model_comparison(
            y_true,
            predictions,
            n_resamples=100,
            random_state=7,
        )
        self.assertEqual(first[2], "strong")
        self.assertTrue(first[0].equals(second[0]))
        self.assertTrue(first[1].equals(second[1]))
        self.assertTrue(first[0]["ci_lower"].between(0.0, 1.0).all())
        weak = first[1].loc[first[1]["model"] == "weak"].iloc[0]
        self.assertLess(weak["point_f1_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
