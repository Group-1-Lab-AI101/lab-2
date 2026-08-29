"""Deployment-oriented analysis and retuning without call duration.

The official Lab 2 experiments intentionally retain every dataset feature. This
module first performs a controlled fixed-configuration ablation, then repeats
the complete training-only hyperparameter search after removing ``duration``.
This is deployment evidence, not a fourth official improvement method.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.artifact_utils import build_experiment_identity, save_json
from src.data import NUMERICAL_FEATURES, PROJECT_ROOT, DatasetSplit, load_and_split_data
from src.evaluation import evaluate_classifier
from src.experiment_contract import (
    CLASS_DISPLAY_NAMES,
    DATASET_PATH,
    FROZEN_BASELINE_PARAMETERS,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.hau_hyperparameter_tuning import (
    COMBINED_SEARCH_SPACE,
    CV_FOLDS,
    make_cv_strategy,
    run_combined_search,
    selected_parameters,
)
from src.preprocessing import build_model_pipeline, get_encoded_feature_names


EXCLUDED_FEATURE = "duration"
PRECALL_NUMERICAL_FEATURES = tuple(
    feature for feature in NUMERICAL_FEATURES if feature != EXCLUDED_FEATURE
)
DEFAULT_FIGURES_OUTPUT = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TREES_OUTPUT = PROJECT_ROOT / "outputs" / "trees"


def build_precall_tree_pipeline(parameters: Mapping[str, Any]) -> Pipeline:
    """Build an unfitted tree pipeline whose inputs cannot include duration."""

    if EXCLUDED_FEATURE in PRECALL_NUMERICAL_FEATURES:
        raise RuntimeError("The pre-call numerical feature set still contains duration.")
    return build_model_pipeline(
        DecisionTreeClassifier(**dict(parameters)),
        numerical_features=PRECALL_NUMERICAL_FEATURES,
    )


def _evaluate_precall_configuration(
    split: DatasetSplit,
    parameters: Mapping[str, Any],
) -> tuple[Pipeline, dict[str, Any], dict[str, int], list[str]]:
    """Fit and evaluate one fixed configuration without the duration column."""

    X_train = split.X_train.drop(columns=[EXCLUDED_FEATURE])
    X_test = split.X_test.drop(columns=[EXCLUDED_FEATURE])
    pipeline = build_precall_tree_pipeline(parameters)
    pipeline.fit(X_train, split.y_train)
    metrics, _, _ = evaluate_classifier(
        pipeline,
        X_train,
        split.y_train,
        X_test,
        split.y_test,
        positive_label=POSITIVE_CLASS,
        positive_class_name=POSITIVE_CLASS_NAME,
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    tree = pipeline.named_steps["model"]
    complexity = {
        "depth": int(tree.get_depth()),
        "leaves": int(tree.get_n_leaves()),
        "nodes": int(tree.tree_.node_count),
    }
    feature_names = get_encoded_feature_names(pipeline)
    if EXCLUDED_FEATURE in feature_names or EXCLUDED_FEATURE in X_train.columns:
        raise RuntimeError("duration reached the fitted pre-call model inputs.")
    return pipeline, metrics, complexity, feature_names


def _comparison_record(
    model: str,
    feature_set: str,
    metrics: Mapping[str, Any],
    complexity: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a sensitivity result for CSV and JSON output."""

    return {
        "model": model,
        "feature_set": feature_set,
        "accuracy": float(metrics["accuracy"]),
        "error_rate": float(metrics["error_rate"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "roc_auc": float(metrics["roc_auc"]),
        "train_accuracy": float(metrics["train_accuracy"]),
        "train_test_accuracy_gap": float(metrics["train_test_accuracy_gap"]),
        "depth": int(complexity["depth"]),
        "leaves": int(complexity["leaves"]),
        "nodes": int(complexity["nodes"]),
    }


def plot_precall_sensitivity(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot predictive changes caused by removing duration at fixed settings."""

    metrics = ("accuracy", "f1_score", "roc_auc")
    positions = np.arange(len(comparison))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12.5, 6.0))
    for offset, metric in enumerate(metrics):
        axis.bar(
            positions + (offset - 1) * width,
            comparison[metric],
            width,
            label=metric.replace("_", " ").title(),
        )
    labels = [
        f"{row.model}\n{row.feature_set}"
        for row in comparison.itertuples(index=False)
    ]
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test score")
    axis.set_title("Deployment Sensitivity: Removing Post-call Duration")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_precall_sensitivity_workflow(
    *,
    baseline_metrics: Mapping[str, Any],
    baseline_complexity: Mapping[str, Any],
    tuned_metrics: Mapping[str, Any],
    tuned_complexity: Mapping[str, Any],
    tuned_parameters: Mapping[str, Any],
    figures_dir: str | Path = DEFAULT_FIGURES_OUTPUT,
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
    search_space: Mapping[str, tuple[Any, ...]] = COMBINED_SEARCH_SPACE,
    cv_folds: int = CV_FOLDS,
    n_jobs: int | None = -1,
) -> dict[str, Any]:
    """Run fixed ablations and a fresh duration-free training-only search."""

    figures = Path(figures_dir).expanduser().resolve()
    results = Path(results_dir).expanduser().resolve()
    trees = Path(trees_dir).expanduser().resolve()
    for directory in (figures, results, trees):
        directory.mkdir(parents=True, exist_ok=True)

    split = load_and_split_data(
        DATASET_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    identity = build_experiment_identity(
        DATASET_PATH,
        split.X_train.index.to_numpy(),
        split.X_test.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        target_mapping=TARGET_MAPPING,
    )

    baseline_pipeline, precall_baseline_metrics, precall_baseline_complexity, feature_names = (
        _evaluate_precall_configuration(split, FROZEN_BASELINE_PARAMETERS)
    )
    tuned_configuration = {"random_state": RANDOM_STATE, **dict(tuned_parameters)}
    fixed_tuned_pipeline, fixed_tuned_metrics, fixed_tuned_complexity, tuned_feature_names = (
        _evaluate_precall_configuration(split, tuned_configuration)
    )
    if feature_names != tuned_feature_names:
        raise RuntimeError("Pre-call configurations produced different feature schemas.")

    X_train_precall = split.X_train.drop(columns=[EXCLUDED_FEATURE])
    X_test_precall = split.X_test.drop(columns=[EXCLUDED_FEATURE])
    search = run_combined_search(
        X_train_precall,
        split.y_train,
        search_space=search_space,
        cv=make_cv_strategy(cv_folds),
        n_jobs=n_jobs,
        numerical_features=PRECALL_NUMERICAL_FEATURES,
    )
    retuned_pipeline = search.best_estimator_
    retuned_metrics, _, _ = evaluate_classifier(
        retuned_pipeline,
        X_train_precall,
        split.y_train,
        X_test_precall,
        split.y_test,
        positive_label=POSITIVE_CLASS,
        positive_class_name=POSITIVE_CLASS_NAME,
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    retuned_tree = retuned_pipeline.named_steps["model"]
    retuned_complexity = {
        "depth": int(retuned_tree.get_depth()),
        "leaves": int(retuned_tree.get_n_leaves()),
        "nodes": int(retuned_tree.tree_.node_count),
    }
    retuned_feature_names = get_encoded_feature_names(retuned_pipeline)
    if EXCLUDED_FEATURE in retuned_feature_names:
        raise RuntimeError("duration reached the freshly retuned pre-call pipeline.")

    comparison = pd.DataFrame(
        [
            _comparison_record(
                "Frozen Baseline", "Post-call (all features)", baseline_metrics, baseline_complexity
            ),
            _comparison_record(
                "Frozen Baseline",
                "Pre-call (no duration)",
                precall_baseline_metrics,
                precall_baseline_complexity,
            ),
            _comparison_record(
                "Hau Tuned", "Post-call (all features)", tuned_metrics, tuned_complexity
            ),
            _comparison_record(
                "Hau Tuned (fixed settings)",
                "Pre-call (no duration)",
                fixed_tuned_metrics,
                fixed_tuned_complexity,
            ),
            _comparison_record(
                "Pre-call Retuned",
                "Pre-call (no duration)",
                retuned_metrics,
                retuned_complexity,
            ),
        ]
    )
    comparison_path = results / "precall_duration_sensitivity.csv"
    comparison.to_csv(comparison_path, index=False)
    figure_path = figures / "precall_duration_sensitivity.png"
    plot_precall_sensitivity(comparison, figure_path)
    joblib.dump(baseline_pipeline, trees / "precall_baseline_pipeline.joblib")
    joblib.dump(fixed_tuned_pipeline, trees / "precall_fixed_hau_pipeline.joblib")
    joblib.dump(retuned_pipeline, trees / "precall_tuned_pipeline.joblib")

    payload = {
        "scope": "Deployment sensitivity and retuning; not an official improvement method",
        "excluded_feature": EXCLUDED_FEATURE,
        "selection_policy": (
            "Fixed configurations isolate feature availability; the pre-call optimum is "
            "then selected by positive-class F1 using stratified training-only CV. The "
            "official test labels never influence hyperparameter selection."
        ),
        "retuned_best_parameters": selected_parameters(search),
        "retuned_best_cv_f1": float(search.best_score_),
        "retuned_cv_folds": int(cv_folds),
        "retuned_candidate_combinations": int(len(search.cv_results_["params"])),
        "raw_predictors": len(split.X_train.columns) - 1,
        "transformed_features": len(feature_names),
        "experiment_identity": identity,
        "comparison": comparison.to_dict(orient="records"),
    }
    payload_path = results / "precall_duration_sensitivity.json"
    save_json(payload, payload_path)
    return {
        "comparison": payload["comparison"],
        "excluded_feature": EXCLUDED_FEATURE,
        "comparison_output": str(comparison_path),
        "summary_output": str(payload_path),
        "figure": str(figure_path),
    }
