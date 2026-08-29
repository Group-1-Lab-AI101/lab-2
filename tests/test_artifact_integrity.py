"""Regression checks for cross-workflow experiment identity validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.artifact_utils import validate_experiment_identity
from src.trung_class_weight import load_team_references


IDENTITY = {
    "dataset_sha256": "dataset-hash",
    "train_indices_sha256": "train-hash",
    "test_indices_sha256": "test-hash",
    "test_size": 0.2,
    "random_state": 42,
    "target_mapping": {"no": 0, "yes": 1},
}
METRICS = {
    "accuracy": 0.9,
    "error_rate": 0.1,
    "precision": 0.6,
    "recall": 0.5,
    "f1_score": 0.55,
    "roc_auc": 0.8,
}
COMPLEXITY = {"depth": 10, "leaves": 20, "nodes": 39}


class ArtifactIntegrityTests(unittest.TestCase):
    """Ensure stale or incomplete reference artifacts cannot be compared."""

    def test_identity_validator_rejects_missing_and_mismatched_records(self) -> None:
        validate_experiment_identity(IDENTITY, IDENTITY, artifact_name="valid")
        with self.assertRaises(RuntimeError):
            validate_experiment_identity(IDENTITY, None, artifact_name="missing")

        stale = {**IDENTITY, "test_indices_sha256": "stale-test-hash"}
        with self.assertRaises(RuntimeError):
            validate_experiment_identity(IDENTITY, stale, artifact_name="stale")

    def test_team_reference_loader_rejects_a_stale_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = root / "results"
            trees = root / "trees"
            results.mkdir()
            trees.mkdir()

            (results / "baseline_metrics.json").write_text(
                json.dumps(METRICS), encoding="utf-8"
            )
            (trees / "tree_analysis.json").write_text(
                json.dumps(COMPLEXITY), encoding="utf-8"
            )
            (results / "run_manifest.json").write_text(
                json.dumps({"experiment_identity": IDENTITY}), encoding="utf-8"
            )
            (results / "hau_best_parameters.json").write_text(
                json.dumps({"experiment_identity": IDENTITY}), encoding="utf-8"
            )
            pd.DataFrame(
                [*METRICS.items(), ("tree_depth", 10), ("number_of_leaves", 20), ("node_count", 39)],
                columns=["Metric", "Result"],
            ).to_csv(results / "hau_tuned_tree_metrics.csv", index=False)
            kiet_payload = {
                "experiment_identity": IDENTITY,
                "selected_criterion": "gini",
                "selected_model": "Pruned Gini",
                "comparison": [{"model": "Pruned Gini", **METRICS, **COMPLEXITY}],
            }
            (results / "kiet_pruning_metrics.json").write_text(
                json.dumps(kiet_payload), encoding="utf-8"
            )

            references = load_team_references(
                results,
                trees,
                expected_identity=IDENTITY,
            )
            self.assertEqual(len(references), 3)

            stale_hau = {**IDENTITY, "dataset_sha256": "different-dataset"}
            (results / "hau_best_parameters.json").write_text(
                json.dumps({"experiment_identity": stale_hau}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                load_team_references(
                    results,
                    trees,
                    expected_identity=IDENTITY,
                )


if __name__ == "__main__":
    unittest.main()
