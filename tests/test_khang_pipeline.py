"""Regression checks for the shared split and preprocessing pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from src.data import load_bank_data, load_and_split_data
from src.eda import build_eda_report
from src.preprocessing import (
    build_model_pipeline,
    build_preprocessing_pipeline,
    get_encoded_feature_names,
)
from src.visualization import generate_eda_figures


class KhangPipelineTests(unittest.TestCase):
    """Verify the shared dataset, EDA, encoding, and model-pipeline contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the dataset and shared split once for all regression checks."""

        cls.frame = load_bank_data()
        cls.split = load_and_split_data()

    def test_dataset_shape_and_schema(self) -> None:
        """Confirm the source file retains its expected shape and completeness."""

        self.assertEqual(self.frame.shape, (45_211, 17))
        self.assertEqual(int(self.frame.isna().sum().sum()), 0)

    def test_eda_report_reproduces_target_counts(self) -> None:
        """Confirm documented target counts are generated from the source data."""

        report = build_eda_report(self.frame)

        self.assertEqual(report["dataset"]["duplicate_rows"], 0)
        self.assertEqual(report["target"]["no"]["count"], 39_922)
        self.assertEqual(report["target"]["yes"]["count"], 5_289)

    def test_split_is_complete_disjoint_and_stratified(self) -> None:
        """Ensure the split is exhaustive, disjoint, and class-stratified."""

        split = self.split
        self.assertEqual(len(split.X_train), 36_168)
        self.assertEqual(len(split.X_test), 9_043)
        self.assertFalse(set(split.X_train.index) & set(split.X_test.index))
        self.assertEqual(len(split.X_train) + len(split.X_test), len(self.frame))
        self.assertAlmostEqual(split.y_train.mean(), split.y_test.mean(), places=3)

    def test_encoding_is_fitted_on_train_and_has_expected_shape(self) -> None:
        """Check encoded dimensions and safe handling of an unseen category."""

        pipeline = build_preprocessing_pipeline()
        train_encoded = pipeline.fit_transform(self.split.X_train)
        test_encoded = pipeline.transform(self.split.X_test)

        self.assertEqual(train_encoded.shape, (36_168, 51))
        self.assertEqual(test_encoded.shape, (9_043, 51))
        self.assertEqual(len(get_encoded_feature_names(pipeline)), 51)

        unseen_category = self.split.X_test.iloc[:1].copy()
        unseen_category.loc[:, "job"] = "new-job-category"
        self.assertEqual(pipeline.transform(unseen_category).shape, (1, 51))

    def test_shared_pipeline_accepts_a_model(self) -> None:
        """Smoke-test fitting and prediction through the reusable pipeline."""

        pipeline = build_model_pipeline(
            DecisionTreeClassifier(max_depth=1, random_state=42)
        )
        pipeline.fit(self.split.X_train, self.split.y_train)
        predictions = pipeline.predict(self.split.X_test.iloc[:10])

        self.assertEqual(predictions.shape, (10,))
        self.assertEqual(set(predictions).difference({0, 1}), set())

    def test_eda_figures_are_written_as_nonempty_pngs(self) -> None:
        """Ensure every documented EDA figure can be generated from source data."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = generate_eda_figures(
                self.frame,
                self.split,
                temporary_directory,
            )

            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertEqual(path.suffix, ".png")
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
