"""Focused regression checks for Kiet's pruning experiment."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.experiment_contract import FROZEN_BASELINE_PARAMETERS
from src.pruning_experiment import (
    DEFAULT_FIGURES_OUTPUT,
    DEFAULT_REPORT,
    DEFAULT_RESULTS_OUTPUT,
    DEFAULT_TREES_OUTPUT,
    make_ccp_alpha_grid,
    sample_ccp_alphas,
    select_best_pruned_candidate,
    tree_parameters,
)


class KietPruningTests(unittest.TestCase):
    def test_parameter_builder_preserves_frozen_baseline(self) -> None:
        before = dict(FROZEN_BASELINE_PARAMETERS)
        parameters = tree_parameters("entropy", 0.001)

        self.assertEqual(parameters["criterion"], "entropy")
        self.assertEqual(parameters["ccp_alpha"], 0.001)
        self.assertEqual(dict(FROZEN_BASELINE_PARAMETERS), before)
        self.assertEqual(FROZEN_BASELINE_PARAMETERS["criterion"], "gini")
        self.assertEqual(FROZEN_BASELINE_PARAMETERS["ccp_alpha"], 0.0)

    def test_alpha_sampling_is_sorted_bounded_and_excludes_stump(self) -> None:
        path = np.linspace(0.0, 0.01, 101)
        sampled = sample_ccp_alphas(path, max_candidates=12)

        self.assertEqual(sampled[0], 0.0)
        self.assertLessEqual(len(sampled), 12)
        self.assertTrue(np.all(np.diff(sampled) > 0.0))
        self.assertLess(sampled[-1], path[-1])

        fixed_grid = make_ccp_alpha_grid(max_candidates=12)
        self.assertEqual(fixed_grid[0], 0.0)
        self.assertEqual(len(fixed_grid), 12)
        self.assertTrue(np.all(np.diff(fixed_grid) > 0.0))

    def test_selection_excludes_unpruned_and_prefers_simpler_ties(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "criterion": "gini",
                    "ccp_alpha": 0.0,
                    "mean_cv_f1": 0.95,
                    "mean_cv_recall": 0.80,
                    "leaves": 100,
                    "nodes": 199,
                },
                {
                    "criterion": "gini",
                    "ccp_alpha": 0.001,
                    "mean_cv_f1": 0.93,
                    "mean_cv_recall": 0.75,
                    "leaves": 30,
                    "nodes": 59,
                },
                {
                    "criterion": "gini",
                    "ccp_alpha": 0.002,
                    "mean_cv_f1": 0.93,
                    "mean_cv_recall": 0.75,
                    "leaves": 20,
                    "nodes": 39,
                },
            ]
        )

        selected = select_best_pruned_candidate(candidates)
        self.assertEqual(selected["ccp_alpha"], 0.002)
        self.assertEqual(selected["leaves"], 20)

    def test_selection_prioritizes_f1_then_recall(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "criterion": "gini",
                    "ccp_alpha": 0.001,
                    "mean_cv_f1": 0.50,
                    "mean_cv_recall": 0.40,
                    "leaves": 10,
                    "nodes": 19,
                },
                {
                    "criterion": "gini",
                    "ccp_alpha": 0.002,
                    "mean_cv_f1": 0.50,
                    "mean_cv_recall": 0.60,
                    "leaves": 20,
                    "nodes": 39,
                },
            ]
        )
        self.assertEqual(select_best_pruned_candidate(candidates)["ccp_alpha"], 0.002)

    def test_invalid_criterion_and_alpha_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tree_parameters("log_loss", 0.0)
        with self.assertRaises(ValueError):
            tree_parameters("gini", -0.1)

    def test_unified_runner_and_output_layout_include_kiet(self) -> None:
        self.assertEqual(DEFAULT_FIGURES_OUTPUT, PROJECT_ROOT / "outputs" / "figures")
        self.assertEqual(DEFAULT_RESULTS_OUTPUT, PROJECT_ROOT / "outputs" / "results")
        self.assertEqual(DEFAULT_TREES_OUTPUT, PROJECT_ROOT / "outputs" / "trees")
        self.assertEqual(
            DEFAULT_REPORT, PROJECT_ROOT / "docs" / "kiet_pruning_and_criterion.md"
        )
        runner = (PROJECT_ROOT / "run_all.py").read_text(encoding="utf-8")
        self.assertIn("run_pruning_workflow", runner)
        self.assertIsInstance(Path(DEFAULT_REPORT), Path)


if __name__ == "__main__":
    unittest.main()
