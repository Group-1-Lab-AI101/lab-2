"""Regression checks for Hau's training-only hyperparameter experiments."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.artifact_utils import index_fingerprint
from src.hau_hyperparameter_tuning import (
    TUNED_PARAMETERS,
    build_combined_parameter_grid,
    build_hau_tree_pipeline,
    evaluate_selected_pipeline,
    load_baseline_reference,
    load_shared_split,
    make_cv_strategy,
    run_combined_search,
    run_hau_hyperparameter_workflow,
    selected_parameters,
)


class HauHyperparameterTuningTests(unittest.TestCase):
    """Verify scope, leakage prevention, outputs, and metric correctness."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.split = load_shared_split()
        sample_indices = cls.split.X_train.index[:3_000]
        cls.X_sample = cls.split.X_train.loc[sample_indices]
        cls.y_sample = cls.split.y_train.loc[sample_indices]
        cls.search = run_combined_search(
            cls.X_sample,
            cls.y_sample,
            search_space={
                "max_depth": (3, 5),
                "min_samples_split": (2,),
                "min_samples_leaf": (1,),
            },
            cv=make_cv_strategy(2),
            n_jobs=1,
        )

    def test_workflow_reuses_the_official_shared_split(self) -> None:
        self.assertEqual(self.split.test_size, 0.20)
        self.assertEqual(self.split.random_state, 42)
        self.assertEqual(len(self.split.X_train), 36_168)
        self.assertEqual(len(self.split.X_test), 9_043)
        self.assertEqual(
            index_fingerprint(self.split.X_train.index.to_numpy()),
            "b078ac348228f8e49c30af463dc1198fb5e8e11d6379e5d842caf580fc728469",
        )
        self.assertEqual(
            index_fingerprint(self.split.X_test.index.to_numpy()),
            "c1c3c7f5f45b5ba6159c45650c2f59b8270c4314b3e403e8a549f9047d292055",
        )

    def test_shared_pipeline_never_learns_a_test_only_category(self) -> None:
        pipeline = build_hau_tree_pipeline()
        pipeline.fit(self.X_sample, self.y_sample)

        test_row = self.split.X_test.iloc[:1].copy()
        test_row.loc[:, "job"] = "hau-test-only-category"
        prediction = pipeline.predict(test_row)
        encoder = pipeline.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ]
        job_categories = set(encoder.categories_[0])

        self.assertEqual(prediction.shape, (1,))
        self.assertNotIn("hau-test-only-category", job_categories)

    def test_search_grid_contains_only_hau_owned_parameters(self) -> None:
        pipeline = build_hau_tree_pipeline()
        grid = build_combined_parameter_grid(pipeline)
        suffixes = {name.split("__", 1)[1] for name in grid}

        self.assertEqual(suffixes, set(TUNED_PARAMETERS))
        self.assertEqual(grid["model__min_samples_split"], [2, 5, 10, 20, 50, 100])
        self.assertEqual(grid["model__min_samples_leaf"], [1, 2, 5, 10, 20, 50])
        self.assertNotIn("model__ccp_alpha", grid)
        self.assertNotIn("model__class_weight", grid)
        estimator = pipeline.named_steps["model"]
        self.assertEqual(estimator.get_params()["ccp_alpha"], 0.0)
        self.assertIsNone(estimator.get_params()["class_weight"])

    def test_combined_search_has_no_test_set_argument(self) -> None:
        parameter_names = set(inspect.signature(run_combined_search).parameters)
        self.assertNotIn("X_test", parameter_names)
        self.assertNotIn("y_test", parameter_names)

    def test_best_pipeline_fits_predicts_and_produces_valid_metrics(self) -> None:
        predictions = self.search.best_estimator_.predict(
            self.split.X_test.iloc[:25]
        )
        metrics, complexity = evaluate_selected_pipeline(
            self.search.best_estimator_,
            self.X_sample,
            self.y_sample,
            self.split.X_test,
            self.split.y_test,
        )

        self.assertEqual(predictions.shape, (25,))
        self.assertEqual(set(predictions).difference({0, 1}), set())
        self.assertEqual(set(selected_parameters(self.search)), set(TUNED_PARAMETERS))
        for name in ("accuracy", "error_rate", "precision", "recall", "f1_score", "roc_auc"):
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)
        self.assertAlmostEqual(metrics["error_rate"], 1.0 - metrics["accuracy"])
        self.assertGreater(complexity["depth"], 0)
        self.assertGreater(complexity["leaves"], 1)
        self.assertGreater(complexity["nodes"], 1)

    def test_reduced_workflow_generates_every_expected_output(self) -> None:
        baseline_metrics, baseline_complexity = load_baseline_reference()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            figures = root / "outputs" / "figures"
            results = root / "outputs" / "results"
            report = root / "docs" / "hau_improvement_methods.md"
            workflow = run_hau_hyperparameter_workflow(
                figures_dir=figures,
                results_dir=results,
                report_path=report,
                baseline_metrics=baseline_metrics,
                baseline_complexity=baseline_complexity,
                single_parameter_candidates={
                    "max_depth": (3,),
                    "min_samples_split": (2,),
                    "min_samples_leaf": (1,),
                },
                combined_search_space={
                    "max_depth": (3,),
                    "min_samples_split": (2,),
                    "min_samples_leaf": (1,),
                },
                cv_folds=2,
                n_jobs=1,
            )

            expected_results = {
                "hau_max_depth_validation.csv",
                "hau_min_samples_split_validation.csv",
                "hau_min_samples_leaf_validation.csv",
                "hau_hyperparameter_search.csv",
                "hau_best_parameters.json",
                "hau_tuned_tree_metrics.csv",
                "hau_baseline_vs_tuned.csv",
            }
            expected_figures = {
                "hau_max_depth_validation.png",
                "hau_min_samples_split_validation.png",
                "hau_min_samples_leaf_validation.png",
                "hau_baseline_vs_tuned.png",
            }
            self.assertTrue(expected_results.issubset({path.name for path in results.iterdir()}))
            self.assertTrue(expected_figures.issubset({path.name for path in figures.iterdir()}))
            self.assertTrue(report.is_file())
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("The single-parameter validation ranges were:", report_text)
            self.assertIn("The combined search ranges were:", report_text)
            self.assertIn("- `max_depth`: `[3]`", report_text)
            for path in figures.iterdir():
                self.assertGreater(path.stat().st_size, 1_000)

            summary = json.loads(
                (results / "hau_best_parameters.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["test_partition_usage"], "one final evaluation after selection")
            self.assertEqual(summary["preprocessing_cv_fit_scope"], "each training fold only")
            self.assertEqual(
                summary["train_indices_sha256"],
                "b078ac348228f8e49c30af463dc1198fb5e8e11d6379e5d842caf580fc728469",
            )
            self.assertEqual(workflow["scoring_metric"], "f1")


if __name__ == "__main__":
    unittest.main()
