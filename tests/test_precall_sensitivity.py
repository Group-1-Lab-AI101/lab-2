"""Regression checks for the duration-free deployment sensitivity analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data import PROJECT_ROOT
from src.precall_sensitivity import run_precall_sensitivity_workflow


class PrecallSensitivityTests(unittest.TestCase):
    """Verify duration exclusion, outputs, and experiment labeling."""

    def test_duration_free_workflow_is_reproducible_and_separate(self) -> None:
        metrics = {
            "accuracy": 0.8,
            "error_rate": 0.2,
            "precision": 0.5,
            "recall": 0.4,
            "f1_score": 0.44,
            "roc_auc": 0.7,
            "train_accuracy": 0.9,
            "train_test_accuracy_gap": 0.1,
        }
        complexity = {"depth": 5, "leaves": 10, "nodes": 19}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workflow = run_precall_sensitivity_workflow(
                baseline_metrics=metrics,
                baseline_complexity=complexity,
                tuned_metrics=metrics,
                tuned_complexity=complexity,
                tuned_parameters={
                    "max_depth": 5,
                    "min_samples_split": 20,
                    "min_samples_leaf": 10,
                },
                figures_dir=root / "figures",
                results_dir=root / "results",
                trees_dir=root / "trees",
                search_space={
                    "max_depth": (5,),
                    "min_samples_split": (20,),
                    "min_samples_leaf": (10,),
                },
                cv_folds=2,
                n_jobs=1,
            )

            summary_path = Path(workflow["summary_output"])
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["excluded_feature"], "duration")
            self.assertEqual(payload["raw_predictors"], 15)
            self.assertEqual(payload["transformed_features"], 50)
            self.assertIn("not an official improvement method", payload["scope"])
            self.assertEqual(len(payload["comparison"]), 5)
            self.assertEqual(payload["retuned_candidate_combinations"], 1)
            self.assertNotIn("duration", payload["retuned_best_parameters"])
            self.assertTrue(Path(workflow["figure"]).is_file())
            self.assertTrue((root / "trees" / "precall_baseline_pipeline.joblib").is_file())
            self.assertTrue((root / "trees" / "precall_fixed_hau_pipeline.joblib").is_file())
            self.assertTrue((root / "trees" / "precall_tuned_pipeline.joblib").is_file())

    def test_runner_includes_sensitivity_and_unused_placeholder_is_removed(self) -> None:
        runner = (PROJECT_ROOT / "run_all.py").read_text(encoding="utf-8")
        self.assertIn("run_precall_sensitivity_workflow", runner)
        self.assertIn('"precall_sensitivity"', runner)
        self.assertFalse((PROJECT_ROOT / "src" / "evaluate.py").exists())
        self.assertTrue((PROJECT_ROOT / "src" / "evaluation.py").is_file())


if __name__ == "__main__":
    unittest.main()
