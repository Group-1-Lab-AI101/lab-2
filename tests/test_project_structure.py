"""Repository-layout and shared-module boundary checks."""

from __future__ import annotations

import unittest

from src.data import PROJECT_ROOT


class ProjectStructureTests(unittest.TestCase):
    def test_all_notebooks_are_consolidated(self) -> None:
        notebooks_dir = PROJECT_ROOT / "notebooks"
        expected = {
            "khang_data_eda_preprocessing.ipynb",
            "hoang_baseline_decision_tree.ipynb",
            "kiet_pruning_visualization.ipynb",
            "trung_class_weight_and_comparison.ipynb",
        }
        actual = {path.name for path in notebooks_dir.glob("*.ipynb")}
        notebooks_outside_directory = list(PROJECT_ROOT.glob("*.ipynb"))
        experiments_dir = PROJECT_ROOT / "experiments"
        if experiments_dir.exists():
            notebooks_outside_directory.extend(experiments_dir.glob("*.ipynb"))

        self.assertEqual(actual, expected)
        self.assertEqual(notebooks_outside_directory, [])

    def test_obsolete_layout_files_are_absent(self) -> None:
        self.assertFalse((PROJECT_ROOT / ".python-version").exists())
        self.assertFalse((PROJECT_ROOT / "outputs" / "figures" / "baseline").exists())
        self.assertFalse((PROJECT_ROOT / "src" / "evaluate.py").exists())

    def test_hoang_artifacts_use_owner_prefix(self) -> None:
        outputs = PROJECT_ROOT / "outputs"
        expected = {
            outputs / "figures" / "hoang_confusion_matrix.png",
            outputs / "figures" / "hoang_baseline_tree_top_levels.png",
            outputs / "figures" / "hoang_baseline_tree_full_structure.png",
            outputs / "figures" / "hoang_top_feature_importance.png",
            outputs / "results" / "hoang_baseline_metrics.json",
            outputs / "results" / "hoang_baseline_metrics.csv",
            outputs / "results" / "hoang_classification_report.csv",
            outputs / "results" / "hoang_feature_importance.csv",
            outputs / "results" / "hoang_preprocessing_audit.json",
            outputs / "results" / "hoang_run_manifest.json",
            outputs / "trees" / "hoang_tree_analysis.json",
            outputs / "trees" / "hoang_early_splits.json",
            outputs / "trees" / "hoang_early_tree.txt",
            outputs / "trees" / "hoang_representative_rules.md",
            outputs / "trees" / "hoang_representative_rules_audit.json",
            outputs / "trees" / "hoang_baseline_tree_full.dot",
            outputs / "trees" / "hoang_baseline_tree_model.joblib",
        }
        self.assertTrue(all(path.is_file() for path in expected))

        obsolete = {
            outputs / "figures" / "confusion_matrix.png",
            outputs / "figures" / "baseline_tree_top_levels.png",
            outputs / "figures" / "baseline_tree_full_structure.png",
            outputs / "figures" / "top_feature_importance.png",
            outputs / "results" / "baseline_metrics.json",
            outputs / "results" / "baseline_metrics.csv",
            outputs / "results" / "classification_report.csv",
            outputs / "results" / "feature_importance.csv",
            outputs / "results" / "preprocessing_audit.json",
            outputs / "results" / "run_manifest.json",
            outputs / "trees" / "tree_analysis.json",
            outputs / "trees" / "early_splits.json",
            outputs / "trees" / "early_tree.txt",
            outputs / "trees" / "representative_rules.md",
            outputs / "trees" / "representative_rules_audit.json",
            outputs / "trees" / "baseline_tree_full.dot",
            outputs / "trees" / "baseline_tree_model.joblib",
        }
        self.assertFalse(any(path.exists() for path in obsolete))

    def test_evaluation_has_one_shared_module(self) -> None:
        evaluation = PROJECT_ROOT / "src" / "evaluation.py"
        self.assertTrue(evaluation.is_file())

        consumers = (
            "baseline_workflow.py",
            "precall_sensitivity.py",
            "pruning_experiment.py",
            "trung_class_weight.py",
        )
        for filename in consumers:
            source = (PROJECT_ROOT / "src" / filename).read_text(encoding="utf-8")
            self.assertIn("from src.evaluation import", source)
            self.assertNotIn("from src.baseline_tree import evaluate_classifier", source)

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("notebooks/hoang_baseline_decision_tree.ipynb", readme)
        self.assertNotIn("notebooks/2-BASELINE-DECISION-TREE.ipynb", readme)
        self.assertNotIn("experiments/2-BASELINE-DECISION-TREE.ipynb", readme)


if __name__ == "__main__":
    unittest.main()
