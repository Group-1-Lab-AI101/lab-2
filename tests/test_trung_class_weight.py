"""Regression checks for Trung's training-only class-weight experiment."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.artifact_utils import index_fingerprint
from src.trung_class_weight import (
    build_trung_tree_pipeline,
    load_shared_split,
    make_cv_strategy,
    run_class_weight_validation,
    run_trung_class_weight_workflow,
    selected_candidate_position,
    select_best_class_weight,
)


class TrungClassWeightTests(unittest.TestCase):
    """Verify parameter scope, leakage prevention, selection, and artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.split = load_shared_split()
        indices = cls.split.X_train.index[:3_000]
        cls.X_sample = cls.split.X_train.loc[indices]
        cls.y_sample = cls.split.y_train.loc[indices]
        cls.candidates = (None, {0: 1.0, 1: 2.0})
        cls.validation = run_class_weight_validation(
            cls.X_sample,
            cls.y_sample,
            candidates=cls.candidates,
            cv=make_cv_strategy(2),
            n_jobs=1,
        )

    def test_workflow_reuses_official_split(self) -> None:
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

    def test_only_class_weight_changes_from_frozen_baseline(self) -> None:
        pipeline = build_trung_tree_pipeline({0: 1.0, 1: 2.0})
        parameters = pipeline.named_steps["model"].get_params(deep=False)
        self.assertEqual(parameters["class_weight"], {0: 1.0, 1: 2.0})
        self.assertIsNone(parameters["max_depth"])
        self.assertEqual(parameters["min_samples_split"], 2)
        self.assertEqual(parameters["min_samples_leaf"], 1)
        self.assertEqual(parameters["ccp_alpha"], 0.0)

    def test_weighting_can_be_combined_with_preselected_structure(self) -> None:
        pipeline = build_trung_tree_pipeline(
            {0: 1.0, 1: 2.0},
            structural_parameters={
                "max_depth": 7,
                "min_samples_split": 20,
                "min_samples_leaf": 5,
            },
        )
        parameters = pipeline.named_steps["model"].get_params(deep=False)
        self.assertEqual(parameters["class_weight"], {0: 1.0, 1: 2.0})
        self.assertEqual(parameters["max_depth"], 7)
        self.assertEqual(parameters["min_samples_split"], 20)
        self.assertEqual(parameters["min_samples_leaf"], 5)

    def test_validation_has_no_test_partition_argument(self) -> None:
        parameters = set(inspect.signature(run_class_weight_validation).parameters)
        self.assertNotIn("X_test", parameters)
        self.assertNotIn("y_test", parameters)

    def test_unified_runner_includes_trung_workflow(self) -> None:
        runner = (Path(__file__).resolve().parents[1] / "run_all.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_trung_class_weight_workflow", runner)
        self.assertIn('"trung_class_weight"', runner)

    def test_validation_and_selection_are_valid(self) -> None:
        selected, record = select_best_class_weight(
            self.validation, self.candidates
        )
        self.assertIn(selected, self.candidates)
        self.assertIn("mean_cv_f1", record)
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertTrue(
                self.validation[f"mean_cv_{metric}"].between(0.0, 1.0).all()
            )

    def test_plot_position_uses_the_selectors_complete_tie_break(self) -> None:
        validation = pd.DataFrame(
            [
                {"candidate_order": 0, "mean_cv_f1": 0.5, "mean_cv_recall": 0.4},
                {"candidate_order": 1, "mean_cv_f1": 0.5, "mean_cv_recall": 0.6},
                {"candidate_order": 2, "mean_cv_f1": 0.4, "mean_cv_recall": 0.8},
            ],
            index=[10, 20, 30],
        )
        candidates = (None, "balanced", {0: 1.0, 1: 2.0})

        _, selected = select_best_class_weight(validation, candidates)
        position = selected_candidate_position(
            validation,
            int(selected["candidate_order"]),
        )

        self.assertEqual(selected["candidate_order"], 1)
        self.assertEqual(position, 1)

    def test_pipeline_ignores_a_test_only_category(self) -> None:
        pipeline = build_trung_tree_pipeline("balanced")
        pipeline.fit(self.X_sample, self.y_sample)
        row = self.split.X_test.iloc[:1].copy()
        row.loc[:, "job"] = "trung-test-only-category"
        prediction = pipeline.predict(row)
        encoder = pipeline.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ]
        self.assertEqual(prediction.shape, (1,))
        self.assertNotIn("trung-test-only-category", set(encoder.categories_[0]))

    def test_reduced_workflow_writes_expected_artifacts(self) -> None:
        references = [
            {
                "Model": "Reference",
                "accuracy": 0.8,
                "error_rate": 0.2,
                "precision": 0.4,
                "recall": 0.3,
                "f1_score": 0.34,
                "roc_auc": 0.7,
                "depth": 5,
                "leaves": 10,
                "nodes": 19,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workflow = run_trung_class_weight_workflow(
                figures_dir=root / "outputs" / "figures",
                results_dir=root / "outputs" / "results",
                trees_dir=root / "outputs" / "trees",
                report_path=root / "docs" / "trung.md",
                candidates=({0: 1.0, 1: 2.0},),
                cv_folds=2,
                n_jobs=1,
                team_references=references,
            )
            expected = (
                root / "outputs" / "results" / "trung_class_weight_validation.csv",
                root / "outputs" / "results" / "trung_best_class_weight.json",
                root / "outputs" / "results" / "trung_weighted_tree_metrics.csv",
                root / "outputs" / "results" / "team_model_comparison.csv",
                root / "outputs" / "figures" / "trung_confusion_matrix.png",
                root / "outputs" / "figures" / "team_model_comparison.png",
                root / "outputs" / "trees" / "trung_weighted_pipeline.joblib",
                root / "docs" / "trung.md",
            )
            self.assertTrue(all(path.is_file() for path in expected))
            summary = json.loads(expected[1].read_text(encoding="utf-8"))
            self.assertEqual(
                summary["test_partition_usage"],
                "one final evaluation after selection",
            )
            self.assertAlmostEqual(
                workflow["metrics"]["error_rate"],
                1.0 - workflow["metrics"]["accuracy"],
            )

    def test_notebook_is_executed_without_errors(self) -> None:
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "trung_class_weight_and_comparison.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        content = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertTrue(all(cell["execution_count"] is not None for cell in code_cells))
        self.assertEqual(errors, [])
        for required in (
            "Trung",
            "class_weight",
            "structural_parameters",
            "Trung Weighted + Tuned",
            "Comparison of Results",
            "Conclusion",
        ):
            self.assertIn(required, content)


if __name__ == "__main__":
    unittest.main()
