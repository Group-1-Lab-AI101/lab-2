"""Regression checks for Hoang's baseline on Khang's shared pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.artifact_utils import index_fingerprint
from src.baseline_workflow import (
    DEFAULT_FIGURES_OUTPUT,
    DEFAULT_REPORT,
    DEFAULT_RESULTS_OUTPUT,
    DEFAULT_SHARED_OUTPUT,
    DEFAULT_TREES_OUTPUT,
)
from src.baseline_tree import (
    baseline_configuration,
    evaluate_classifier,
    extract_feature_importance,
    train_baseline_tree,
)
from src.data import PROJECT_ROOT, load_and_split_data
from src.experiment_contract import (
    CLASS_DISPLAY_NAMES,
    FROZEN_BASELINE_PARAMETERS,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names


class BaselineTests(unittest.TestCase):
    """Ensure integration does not change the shared split or frozen model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.split = load_and_split_data(
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        cls.preprocessing = build_preprocessing_pipeline()
        cls.X_train_processed = cls.preprocessing.fit_transform(cls.split.X_train)
        cls.X_test_processed = cls.preprocessing.transform(cls.split.X_test)
        cls.feature_names = np.asarray(
            get_encoded_feature_names(cls.preprocessing), dtype=str
        )

    def test_shared_split_identity_and_target_exclusion(self) -> None:
        self.assertEqual(len(self.split.X_train), 36_168)
        self.assertEqual(len(self.split.X_test), 9_043)
        self.assertFalse(set(self.split.X_train.index) & set(self.split.X_test.index))
        self.assertNotIn("y", self.split.X_train.columns)
        self.assertNotIn("y", self.split.X_test.columns)
        self.assertEqual(
            index_fingerprint(self.split.X_train.index.to_numpy()),
            "b078ac348228f8e49c30af463dc1198fb5e8e11d6379e5d842caf580fc728469",
        )
        self.assertEqual(
            index_fingerprint(self.split.X_test.index.to_numpy()),
            "c1c3c7f5f45b5ba6159c45650c2f59b8270c4314b3e403e8a549f9047d292055",
        )

    def test_preprocessing_and_feature_names_align(self) -> None:
        self.assertEqual(self.X_train_processed.shape, (36_168, 51))
        self.assertEqual(self.X_test_processed.shape, (9_043, 51))
        self.assertEqual(len(self.feature_names), 51)
        self.assertNotIn("y", self.feature_names)

    def test_baseline_evaluation_uses_probabilities(self) -> None:
        model = train_baseline_tree(self.X_train_processed, self.split.y_train)
        with patch.object(model, "predict_proba", wraps=model.predict_proba) as probabilities:
            metrics, report, predictions = evaluate_classifier(
                model,
                self.X_train_processed,
                self.split.y_train,
                self.X_test_processed,
                self.split.y_test,
                positive_label=POSITIVE_CLASS,
                positive_class_name=POSITIVE_CLASS_NAME,
                class_display_names=CLASS_DISPLAY_NAMES,
            )
            probabilities.assert_called_once()

        importance = extract_feature_importance(model, self.feature_names)
        self.assertEqual(len(predictions), len(self.split.y_test))
        self.assertIn("yes", report.index)
        self.assertTrue(np.isnan(report.loc["accuracy", "precision"]))
        self.assertTrue(np.isnan(report.loc["accuracy", "recall"]))
        self.assertAlmostEqual(
            report.loc["accuracy", "f1-score"], metrics["accuracy"]
        )
        self.assertEqual(report.loc["accuracy", "support"], len(self.split.y_test))
        self.assertEqual(metrics["positive_class"], "yes")
        self.assertEqual(metrics["positive_class_encoded_value"], 1)
        majority_accuracy = self.split.y_test.value_counts(normalize=True).max()
        self.assertLess(metrics["accuracy"], majority_accuracy)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(len(importance), self.X_train_processed.shape[1])
        self.assertAlmostEqual(importance["Importance"].sum(), 1.0)
        self.assertEqual(baseline_configuration(model), dict(FROZEN_BASELINE_PARAMETERS))

    def test_frozen_baseline_mapping_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            FROZEN_BASELINE_PARAMETERS["random_state"] = 7  # type: ignore[index]

    def test_unified_runner_and_output_layout(self) -> None:
        self.assertEqual(DEFAULT_FIGURES_OUTPUT, PROJECT_ROOT / "outputs" / "figures")
        self.assertEqual(DEFAULT_RESULTS_OUTPUT, PROJECT_ROOT / "outputs" / "results")
        self.assertEqual(DEFAULT_TREES_OUTPUT, PROJECT_ROOT / "outputs" / "trees")
        self.assertEqual(DEFAULT_SHARED_OUTPUT, PROJECT_ROOT / "outputs" / "shared")
        self.assertEqual(
            DEFAULT_REPORT,
            PROJECT_ROOT / "docs" / "hoang_baseline_and_tree_analysis.md",
        )
        self.assertTrue((PROJECT_ROOT / "run_all.py").is_file())
        self.assertFalse((PROJECT_ROOT / "run_baseline.py").exists())
        self.assertFalse((PROJECT_ROOT / "artifacts").exists())
        self.assertFalse((PROJECT_ROOT / "reports").exists())
        self.assertFalse((PROJECT_ROOT / "outputs" / "baseline").exists())
        self.assertIsInstance(Path(DEFAULT_RESULTS_OUTPUT), Path)


if __name__ == "__main__":
    unittest.main()
