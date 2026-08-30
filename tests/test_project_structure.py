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
