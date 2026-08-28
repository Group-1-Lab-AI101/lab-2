"""Focused tests for Hoang's baseline-only functions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.bank_data import prepare_bank_data, preprocessing_audit
from src.baseline_tree import (
    baseline_configuration,
    evaluate_classifier,
    extract_feature_importance,
    train_baseline_tree,
)
from src.experiment_contract import FROZEN_BASELINE_PARAMETERS, RANDOM_STATE, TEST_SIZE


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "age": list(range(20, 60)),
                "job": ["admin", "student", "technician", "services"] * 10,
                "duration": [value * 11 % 300 for value in range(40)],
                "y": ["no", "no", "yes", "no"] * 10,
            }
        )

    def test_preprocessing_excludes_target_and_aligns_names(self) -> None:
        prepared = prepare_bank_data(self.frame)
        audit = preprocessing_audit(prepared)
        self.assertTrue(audit["target_excluded_from_raw_features"])
        self.assertTrue(audit["target_excluded_from_transformed_features"])
        self.assertTrue(audit["feature_names_aligned"])
        self.assertTrue(audit["train_test_indices_disjoint"])
        self.assertNotIn("y", prepared.feature_names)
        self.assertNotIn("y", prepared.X_train_raw.columns)
        self.assertNotIn("y", prepared.X_test_raw.columns)
        self.assertEqual(len(prepared.y_test), int(len(self.frame) * TEST_SIZE))

    def test_baseline_evaluation_and_importance_are_consistent(self) -> None:
        prepared = prepare_bank_data(self.frame)
        model = train_baseline_tree(prepared.X_train_processed, prepared.y_train)
        with patch.object(model, "predict_proba", wraps=model.predict_proba) as probabilities:
            metrics, report, predictions = evaluate_classifier(
                model,
                prepared.X_train_processed,
                prepared.y_train,
                prepared.X_test_processed,
                prepared.y_test,
                positive_label="yes",
            )
            probabilities.assert_called_once()
        importance = extract_feature_importance(model, prepared.feature_names)
        self.assertEqual(len(predictions), len(prepared.y_test))
        self.assertIn("yes", report.index)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(len(importance), prepared.X_train_processed.shape[1])
        self.assertAlmostEqual(importance["Importance"].sum(), 1.0)
        self.assertEqual(baseline_configuration(model), dict(FROZEN_BASELINE_PARAMETERS))
        self.assertEqual(model.random_state, RANDOM_STATE)

    def test_shared_split_is_exactly_reproducible(self) -> None:
        first = prepare_bank_data(self.frame)
        second = prepare_bank_data(self.frame)
        self.assertListEqual(
            first.train_row_indices.tolist(), second.train_row_indices.tolist()
        )
        self.assertListEqual(
            first.test_row_indices.tolist(), second.test_row_indices.tolist()
        )
        self.assertListEqual(first.y_test.tolist(), second.y_test.tolist())

    def test_frozen_baseline_mapping_is_immutable(self) -> None:
        with self.assertRaises(TypeError):
            FROZEN_BASELINE_PARAMETERS["random_state"] = 7  # type: ignore[index]

    def test_encoder_learns_categories_from_training_rows_only(self) -> None:
        frame = pd.DataFrame(
            {
                "value": list(range(100)),
                "unique_category": [f"category_{index}" for index in range(100)],
                "y": ["no", "yes"] * 50,
            }
        )
        prepared = prepare_bank_data(frame)
        learned = set(prepared.preprocessor.named_transformers_["categorical"].categories_[0])
        train_categories = set(prepared.X_train_raw["unique_category"])
        test_categories = set(prepared.X_test_raw["unique_category"])
        self.assertSetEqual(learned, train_categories)
        self.assertTrue(test_categories.isdisjoint(learned))


if __name__ == "__main__":
    unittest.main()
