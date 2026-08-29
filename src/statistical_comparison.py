"""Paired stratified bootstrap uncertainty for the official held-out models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.artifact_utils import save_json
from src.data import PROJECT_ROOT, load_and_split_data
from src.experiment_contract import RANDOM_STATE, TEST_SIZE


DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TREES_OUTPUT = PROJECT_ROOT / "outputs" / "trees"
DEFAULT_SHARED_OUTPUT = PROJECT_ROOT / "outputs" / "shared"
BOOTSTRAP_REPLICATES = 2_000
CONFIDENCE_LEVEL = 0.95


def stratified_bootstrap_indices(
    y_true: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_REPLICATES,
    random_state: int = RANDOM_STATE,
) -> list[np.ndarray]:
    """Sample each class with replacement so every replicate keeps both classes."""

    labels = np.unique(y_true)
    if labels.size != 2:
        raise ValueError("The bootstrap requires exactly two target classes.")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")
    class_indices = [np.flatnonzero(y_true == label) for label in labels]
    if any(indices.size == 0 for indices in class_indices):
        raise ValueError("Every target class must contain at least one observation.")
    generator = np.random.default_rng(random_state)
    return [
        np.concatenate(
            [generator.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        for _ in range(n_resamples)
    ]


def bootstrap_model_comparison(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    n_resamples: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return percentile CIs and paired F1 differences for common test rows."""

    y = np.asarray(y_true)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    if not predictions:
        raise ValueError("At least one model prediction vector is required.")
    normalized = {name: np.asarray(values) for name, values in predictions.items()}
    if any(values.shape != y.shape for values in normalized.values()):
        raise ValueError("Every prediction vector must align with y_true.")

    samples = stratified_bootstrap_indices(
        y,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    metric_samples: dict[str, dict[str, np.ndarray]] = {}
    point_f1: dict[str, float] = {}
    for name, predicted in normalized.items():
        point_f1[name] = float(f1_score(y, predicted, pos_label=1, zero_division=0))
        metric_samples[name] = {
            "accuracy": np.asarray(
                [accuracy_score(y[index], predicted[index]) for index in samples]
            ),
            "f1_score": np.asarray(
                [
                    f1_score(y[index], predicted[index], pos_label=1, zero_division=0)
                    for index in samples
                ]
            ),
        }

    alpha = (1.0 - confidence_level) / 2.0
    interval_rows: list[dict[str, Any]] = []
    for name, metrics in metric_samples.items():
        for metric, values in metrics.items():
            point = (
                accuracy_score(y, normalized[name])
                if metric == "accuracy"
                else point_f1[name]
            )
            interval_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "point_estimate": float(point),
                    "ci_lower": float(np.quantile(values, alpha)),
                    "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                    "confidence_level": float(confidence_level),
                    "bootstrap_replicates": int(n_resamples),
                }
            )

    reference = max(point_f1, key=point_f1.get)
    reference_samples = metric_samples[reference]["f1_score"]
    difference_rows = []
    for name in normalized:
        difference = metric_samples[name]["f1_score"] - reference_samples
        difference_rows.append(
            {
                "model": name,
                "reference_model": reference,
                "point_f1_difference": float(point_f1[name] - point_f1[reference]),
                "ci_lower": float(np.quantile(difference, alpha)),
                "ci_upper": float(np.quantile(difference, 1.0 - alpha)),
                "interval_includes_zero": bool(
                    np.quantile(difference, alpha) <= 0.0 <= np.quantile(difference, 1.0 - alpha)
                ),
            }
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(difference_rows), reference


def run_statistical_comparison_workflow(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
    shared_dir: str | Path = DEFAULT_SHARED_OUTPUT,
    n_resamples: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Load official fitted artifacts and persist paired-bootstrap evidence."""

    results = Path(results_dir).resolve()
    trees = Path(trees_dir).resolve()
    shared = Path(shared_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    split = load_and_split_data(test_size=TEST_SIZE, random_state=RANDOM_STATE)

    preprocessor = joblib.load(shared / "preprocessor.joblib")
    X_test_processed = preprocessor.transform(split.X_test)
    kiet_summary = json.loads(
        (results / "kiet_pruning_metrics.json").read_text(encoding="utf-8")
    )
    kiet_name = f"Kiet {kiet_summary['selected_model']}"
    models = {
        "Hoang Baseline": (joblib.load(trees / "baseline_tree_model.joblib"), X_test_processed),
        "Hau Tuned": (joblib.load(trees / "hau_tuned_pipeline.joblib"), split.X_test),
        kiet_name: (joblib.load(trees / "kiet_pruned_tree_model.joblib"), X_test_processed),
        "Trung Weighted + Tuned": (
            joblib.load(trees / "trung_weighted_pipeline.joblib"),
            split.X_test,
        ),
    }
    predictions = {name: model.predict(features) for name, (model, features) in models.items()}
    intervals, differences, reference = bootstrap_model_comparison(
        split.y_test.to_numpy(),
        predictions,
        n_resamples=n_resamples,
    )
    intervals_path = results / "bootstrap_confidence_intervals.csv"
    differences_path = results / "paired_bootstrap_f1_differences.csv"
    intervals.to_csv(intervals_path, index=False)
    differences.to_csv(differences_path, index=False)
    payload = {
        "method": "paired stratified percentile bootstrap on the common held-out rows",
        "confidence_level": CONFIDENCE_LEVEL,
        "bootstrap_replicates": int(n_resamples),
        "descriptive_reference_highest_test_f1": reference,
        "selection_warning": (
            "Intervals quantify held-out uncertainty and do not replace training-only "
            "hyperparameter selection."
        ),
        "intervals": intervals.to_dict(orient="records"),
        "paired_f1_differences": differences.to_dict(orient="records"),
    }
    summary_path = results / "bootstrap_uncertainty.json"
    save_json(payload, summary_path)
    return {
        "reference_model": reference,
        "intervals": payload["intervals"],
        "paired_f1_differences": payload["paired_f1_differences"],
        "intervals_output": str(intervals_path),
        "differences_output": str(differences_path),
        "summary_output": str(summary_path),
    }
