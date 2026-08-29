"""Orchestrate Hoang's baseline workflow for the unified project runner."""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Any

# Matplotlib reads this setting at import time. Use writable temporary state,
# keeping environment caches out of the repository and avoiding host warnings.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from src.artifact_utils import build_experiment_identity, save_json
from src.baseline_tree import (
    audit_representative_rules,
    baseline_configuration,
    export_full_tree_dot,
    extract_early_splits,
    extract_early_tree_text,
    extract_feature_importance,
    extract_representative_rules,
    extract_tree_statistics,
    plot_feature_importance,
    plot_full_tree_structure,
    plot_tree_top_levels,
    train_baseline_tree,
    write_representative_rules,
)
from src.evaluation import evaluate_classifier, plot_confusion_matrix
from src.data import PROJECT_ROOT, load_and_split_data
from src.experiment_contract import (
    CLASS_DISPLAY_NAMES,
    DATASET_DISPLAY_PATH,
    DATASET_PATH,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    SPLIT_STRATEGY,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names


DEFAULT_FIGURES_OUTPUT = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TREES_OUTPUT = PROJECT_ROOT / "outputs" / "trees"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "hoang_baseline_and_tree_analysis.md"
DEFAULT_SHARED_OUTPUT = PROJECT_ROOT / "outputs" / "shared"


def build_preprocessing_audit(
    split: Any,
    X_train_processed: Any,
    X_test_processed: Any,
    feature_names: np.ndarray,
) -> dict[str, Any]:
    """Record leakage-prevention and feature-alignment checks.

    Args:
        split: Shared train/test partition containing raw predictors and targets.
        X_train_processed: Training matrix produced by the fitted preprocessor.
        X_test_processed: Test matrix transformed without refitting.
        feature_names: Ordered names corresponding to transformed matrix columns.

    Returns:
        A JSON-compatible audit dictionary covering target exclusion, disjoint
        indices, fit scope, row counts, and transformed-feature alignment.
    """

    train_indices = split.X_train.index.to_numpy(dtype=np.int64, copy=True)
    test_indices = split.X_test.index.to_numpy(dtype=np.int64, copy=True)
    return {
        "shared_pipeline": "src.data + src.preprocessing",
        "split_before_preprocessor_fit": True,
        "preprocessor_fit_partition": "training only",
        "test_partition_usage": "transform and evaluation only",
        "target_column": TARGET_COLUMN,
        "target_excluded_from_raw_partitions": TARGET_COLUMN
        not in split.X_train.columns
        and TARGET_COLUMN not in split.X_test.columns,
        "target_excluded_from_transformed_features": TARGET_COLUMN
        not in set(feature_names),
        "train_test_indices_disjoint": bool(
            np.intersect1d(train_indices, test_indices).size == 0
        ),
        "split_row_count": int(len(train_indices) + len(test_indices)),
        "feature_name_count": int(len(feature_names)),
        "transformed_train_column_count": int(X_train_processed.shape[1]),
        "transformed_test_column_count": int(X_test_processed.shape[1]),
        "feature_names_aligned": bool(
            X_train_processed.shape[1]
            == X_test_processed.shape[1]
            == len(feature_names)
        ),
    }


def render_baseline_report(
    *,
    report_path: Path,
    figures_dir: Path,
    trees_dir: Path,
    dataset_path: Path,
    row_count: int,
    raw_feature_count: int,
    transformed_feature_count: int,
    test_size: float,
    random_state: int,
    configuration: dict[str, Any],
    metrics: dict[str, Any],
    statistics: dict[str, Any],
    importance: pd.DataFrame,
    early_splits: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    leakage_audit: dict[str, Any],
) -> None:
    """Render the measured baseline results as report-ready Markdown.

    Args:
        report_path: Destination Markdown path under the unified ``docs`` tree.
        figures_dir: Directory containing baseline PNG visualizations.
        trees_dir: Directory containing the fitted model and tree exports.
        dataset_path: Display path of the source dataset.
        row_count: Total number of observations in train and test.
        raw_feature_count: Number of predictor columns before preprocessing.
        transformed_feature_count: Number of encoded output features.
        test_size: Held-out fraction used by the frozen experiment contract.
        random_state: Seed used for the shared stratified split.
        configuration: Exact fitted Decision Tree constructor parameters.
        metrics: Measured train/test classification metrics.
        statistics: Measured tree depth, size, and early-split information.
        importance: Ranked transformed-feature importance table.
        early_splits: Structured description of the first tree levels.
        rules: Representative fitted leaf rules selected programmatically.
        leakage_audit: Recorded preprocessing and partition safety checks.

    Returns:
        None. The function writes UTF-8 Markdown to ``report_path``.

    Raises:
        OSError: If the report directory or file cannot be written.
    """

    relative_figures = Path(
        os.path.relpath(figures_dir.resolve(), start=report_path.parent.resolve())
    )
    relative_trees = Path(
        os.path.relpath(trees_dir.resolve(), start=report_path.parent.resolve())
    )
    matrix = metrics["confusion_matrix"]
    class_support = [sum(row) for row in matrix]
    test_observations = sum(class_support)
    majority_class_index = int(np.argmax(class_support))
    majority_class_name = metrics["class_labels"][majority_class_index]
    majority_accuracy = class_support[majority_class_index] / test_observations
    majority_accuracy_gap = metrics["accuracy"] - majority_accuracy
    top_features = importance.head(10)
    config_text = ", ".join(f"`{key}={value!r}`" for key, value in configuration.items())
    feature_rows = "\n".join(
        f"| {int(row.Rank)} | `{row.Feature}` | {row.Importance:.6f} |"
        for row in top_features.itertuples(index=False)
    )
    early_split_rows = "\n".join(
        (
            f"| {split['depth']} | `{split['branch_path']}` | `{split['feature']}` | "
            f"{split['threshold']:.3f} | {split['training_samples']:,} | "
            f"`{split['predicted_class']}` |"
        )
        for split in early_splits
    )
    rule_sections: list[str] = []
    for index, rule in enumerate(rules, start=1):
        conditions = " AND ".join(rule["conditions"])
        rule_sections.append(
            f"{index}. IF {conditions}, THEN predict **{rule['predicted_class']}** "
            f"(training support = {rule['sample_count']:,}, purity = {rule['purity']:.3f}, "
            f"held-out support = {rule['evaluation_sample_count']:,}, held-out purity = "
            + (
                f"{rule['evaluation_purity']:.3f}, "
                if rule["evaluation_purity"] is not None
                else "not estimable, "
            )
            + f"path depth = {rule['depth']})."
        )
    leakage_checks = "\n".join(
        f"- {key.replace('_', ' ').capitalize()}: `{value}`"
        for key, value in leakage_audit.items()
    )
    overfit_statement = (
        "The baseline shows strong evidence of overfitting"
        if metrics["train_test_accuracy_gap"] >= 0.05 and metrics["train_accuracy"] >= 0.99
        else "The baseline does not show a large measured train-test gap"
    )

    report = f"""# Baseline Model

## Baseline Configuration

This section reports only Hoang's baseline Decision Tree and analysis. The source file was `{dataset_path}` with {row_count:,} observations, {raw_feature_count} raw predictors, and the binary target `y`. The positive class is `{metrics['positive_class']}`, meaning that the client subscribed to a term deposit.

Hoang's baseline directly reuses Khang's shared data and preprocessing modules: `src.data.load_and_split_data()` performs a stratified {(1-test_size):.0%}/{test_size:.0%} train/test split with `random_state={random_state}`, and `src.preprocessing.build_preprocessing_pipeline()` applies one-hot encoding to categorical predictors with numerical passthrough. The target is mapped as `no=0` and `yes=1`; report labels use the original class names. The encoder was fitted on the training partition only and produced {transformed_feature_count} correctly named transformed features.

The true untuned baseline was `DecisionTreeClassifier` with the following exact configuration: {config_text}. No hyperparameter search, pruning, class weighting, or cross-validation tuning was performed.

## Baseline Evaluation

All evaluation values below were computed from the untouched held-out test partition. Precision, recall, and F1-score refer to positive class `{metrics['positive_class']}`. ROC-AUC uses `predict_proba` scores for that class. Undefined metric divisions, if any, are assigned zero (`zero_division=0`); none of the displayed positive-class metrics required fabrication or imputation.

| Metric | Result |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.6f} |
| Error Rate | {metrics['error_rate']:.6f} |
| Precision (`{metrics['positive_class']}`) | {metrics['precision']:.6f} |
| Recall (`{metrics['positive_class']}`) | {metrics['recall']:.6f} |
| F1-score (`{metrics['positive_class']}`) | {metrics['f1_score']:.6f} |
| ROC-AUC | {metrics['roc_auc']:.6f} |
| Training Accuracy | {metrics['train_accuracy']:.6f} |
| Test Accuracy | {metrics['test_accuracy']:.6f} |
| Train-Test Accuracy Gap | {metrics['train_test_accuracy_gap']:.6f} |

Accuracy is the overall fraction classified correctly. Precision measures how often predicted subscriptions were correct; recall measures how many actual subscriptions were detected; F1 balances precision and recall. ROC-AUC measures ranking discrimination across thresholds, while the error rate is the fraction classified incorrectly.

Because the held-out set is imbalanced, an always-`{majority_class_name}` reference would achieve accuracy {majority_accuracy:.6f}. The baseline accuracy is {abs(majority_accuracy_gap):.6f} {'higher' if majority_accuracy_gap >= 0 else 'lower'} than that reference. The always-majority rule would detect no positive subscriptions, whereas the fitted baseline identifies {matrix[1][1]} true positives; therefore accuracy must be interpreted together with positive-class recall, F1-score, and the confusion matrix. This is evaluation context, not an additional improvement experiment.

## Confusion Matrix

![Baseline confusion matrix]({relative_figures / 'confusion_matrix.png'})

The class order is `{metrics['class_labels'][0]}`, `{metrics['class_labels'][1]}`. Therefore, the matrix is `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]`: {matrix[0][0]} true negatives, {matrix[0][1]} false positives, {matrix[1][0]} false negatives, and {matrix[1][1]} true positives. The horizontal axis is the predicted label and the vertical axis is the true label.

# Analysis of the Tree

## Tree Structure

The fitted baseline has depth **{statistics['depth']}**, **{statistics['nodes']:,} nodes**, and **{statistics['leaves']:,} leaves**. Its training accuracy is {metrics['train_accuracy']:.6f}, versus {metrics['test_accuracy']:.6f} on the test set, a gap of {metrics['train_test_accuracy_gap']:.6f}. {overfit_statement}: the unrestricted tree fits the training observations{' perfectly' if metrics['train_accuracy'] == 1.0 else ' very closely'} but generalizes substantially less accurately. This diagnosis describes the measured baseline and is not the result of tuning.

![Full baseline tree structure]({relative_figures / 'baseline_tree_full_structure.png'})

The full structural view contains every node, with colour indicating the node's predicted class. Labels are intentionally omitted at this scale. The following unchanged-model view displays the first levels with readable node labels; it is a presentation truncation, not a smaller retrained tree.

![Baseline tree top levels]({relative_figures / 'baseline_tree_top_levels.png'})

## Important Splits and Decision Rules

The root split uses `{statistics['root_split_feature']}` at threshold {statistics['root_split_threshold']:.6f}. The left branch represents **{statistics['root_left_condition']}**, and the right branch represents **{statistics['root_right_condition']}**. The threshold is interpreted as {statistics['root_split_kind']}.

The internal splits through depth 2 are listed below. For a one-hot feature such as `poutcome_success`, a threshold of 0.5 separates rows without that category (left) from rows with that category (right).

| Depth | Branch path | Split feature | Threshold | Training rows at node | Node prediction |
| ---: | --- | --- | ---: | ---: | --- |
{early_split_rows}

Representative fitted leaf rules were selected programmatically for high support and moderate path length:

{chr(10).join(rule_sections)}

These rules describe associations learned by this fitted tree. They should not be interpreted as causal effects. The exact early-level tree text is saved in `{relative_trees / 'early_tree.txt'}`, and the complete labeled tree is available in `{relative_trees / 'baseline_tree_full.dot'}`.

## Feature Importance

The tree's impurity-based `feature_importances_` values were mapped one-to-one to fitted `ColumnTransformer.get_feature_names_out()` names.

| Rank | Feature | Importance |
| ---: | --- | ---: |
{feature_rows}

![Top feature importances]({relative_figures / 'top_feature_importance.png'})

An importance value is the normalized total impurity reduction attributed to a transformed feature. It indicates how much the fitted tree used that feature, but it does not establish causality and may favour variables offering many possible split points.

## Strengths of the Baseline Tree

- It represents nonlinear interactions without requiring feature scaling.
- Its fitted decisions can be inspected through explicit branches and leaf rules.
- It provides a reproducible reference result for the separately owned improvement experiments.

## Weaknesses of the Baseline Tree

- Its depth, leaf count, and measured train-test gap make the unrestricted baseline difficult to interpret in full and indicate poor generalization relative to its training fit.
- Its accuracy is below the always-majority reference on this imbalanced test set, and positive-class recall/F1 remain weak; accuracy alone would therefore be misleading.
- Impurity-based feature importance is model-specific and non-causal.
- The `duration` predictor is only known after a marketing call finishes. Its use is valid for reproducing the selected dataset baseline, but it would be unavailable for a pre-call targeting system and is therefore a deployment-time leakage concern, not a train/test leakage bug.

## Data-Leakage Audit

{leakage_checks}

The test partition was never supplied to model or preprocessing `fit`, and the target was removed before splitting predictors. No test result was used to tune this baseline. Khang's original train/test indices are fingerprinted in the shared manifest so every later experiment can verify direct comparability.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_baseline_workflow(
    *,
    figures_dir: str | Path = DEFAULT_FIGURES_OUTPUT,
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    shared_output_dir: str | Path = DEFAULT_SHARED_OUTPUT,
) -> dict[str, Any]:
    """Train, evaluate, document, and persist the frozen baseline tree.

    Args:
        figures_dir: Unified directory for confusion matrix, feature importance,
            and baseline tree PNG visualizations.
        results_dir: Unified directory for metrics, tabular reports, audits, and
            reproducibility metadata.
        trees_dir: Unified directory for the fitted model, DOT, rules, structural
            statistics, and textual tree exports.
        report_path: Markdown destination for Hoang's baseline analysis.
        shared_output_dir: Unified output directory for the fitted preprocessor,
            exact train/test indices, and shared split manifest.

    Returns:
        A compact JSON-compatible summary containing measured metrics, tree
        statistics, the top ten transformed features, and all output locations.

    Raises:
        RuntimeError: If preprocessing leakage or feature-alignment checks fail.
        OSError: If any output, report, plot, or serialized artifact cannot be
            written.
        ValueError: If the shared dataset or frozen experiment contract is invalid.
    """

    resolved_figures = Path(figures_dir).expanduser().resolve()
    resolved_results = Path(results_dir).expanduser().resolve()
    resolved_trees = Path(trees_dir).expanduser().resolve()
    resolved_report = Path(report_path).expanduser().resolve()
    resolved_shared_output = Path(shared_output_dir).expanduser().resolve()
    resolved_figures.mkdir(parents=True, exist_ok=True)
    resolved_results.mkdir(parents=True, exist_ok=True)
    resolved_trees.mkdir(parents=True, exist_ok=True)
    resolved_shared_output.mkdir(parents=True, exist_ok=True)

    split = load_and_split_data(
        DATASET_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    preprocessing_pipeline = build_preprocessing_pipeline()
    X_train_processed = preprocessing_pipeline.fit_transform(split.X_train)
    X_test_processed = preprocessing_pipeline.transform(split.X_test)
    feature_names = np.asarray(
        get_encoded_feature_names(preprocessing_pipeline), dtype=str
    )
    train_row_indices = split.X_train.index.to_numpy(dtype=np.int64, copy=True)
    test_row_indices = split.X_test.index.to_numpy(dtype=np.int64, copy=True)
    audit = build_preprocessing_audit(
        split,
        X_train_processed,
        X_test_processed,
        feature_names,
    )
    if not all(
        audit[key]
        for key in (
            "target_excluded_from_transformed_features",
            "target_excluded_from_raw_partitions",
            "train_test_indices_disjoint",
            "feature_names_aligned",
        )
    ):
        raise RuntimeError(f"Preprocessing audit failed: {audit}")

    model = train_baseline_tree(
        X_train_processed,
        split.y_train,
    )
    metrics, classification_report_frame, y_test_pred = evaluate_classifier(
        model,
        X_train_processed,
        split.y_train,
        X_test_processed,
        split.y_test,
        positive_label=POSITIVE_CLASS,
        positive_class_name=POSITIVE_CLASS_NAME,
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    configuration = baseline_configuration(model)
    importance = extract_feature_importance(model, feature_names)
    statistics = extract_tree_statistics(model, feature_names, metrics)
    early_splits = extract_early_splits(
        model, feature_names, class_display_names=CLASS_DISPLAY_NAMES
    )
    rules = extract_representative_rules(
        model, feature_names, class_display_names=CLASS_DISPLAY_NAMES
    )
    rules = audit_representative_rules(
        model,
        rules,
        X_test_processed,
        split.y_test,
    )

    save_json(metrics, resolved_results / "baseline_metrics.json")
    pd.DataFrame(
        [(key, value) for key, value in metrics.items() if isinstance(value, (int, float, str))],
        columns=["Metric", "Result"],
    ).to_csv(resolved_results / "baseline_metrics.csv", index=False)
    classification_report_frame.to_csv(
        resolved_results / "classification_report.csv"
    )
    importance.to_csv(resolved_results / "feature_importance.csv", index=False)
    save_json(statistics, resolved_trees / "tree_analysis.json")
    save_json({"splits": early_splits}, resolved_trees / "early_splits.json")
    save_json(audit, resolved_results / "preprocessing_audit.json")
    (resolved_trees / "early_tree.txt").write_text(
        extract_early_tree_text(model, feature_names), encoding="utf-8"
    )
    write_representative_rules(rules, resolved_trees / "representative_rules.md")
    save_json({"rules": rules}, resolved_trees / "representative_rules_audit.json")

    plot_confusion_matrix(
        split.y_test,
        y_test_pred,
        model.classes_,
        resolved_figures / "confusion_matrix.png",
        class_display_names=CLASS_DISPLAY_NAMES,
        title="Baseline Decision Tree - Confusion Matrix",
    )
    plot_tree_top_levels(
        model,
        feature_names,
        resolved_figures / "baseline_tree_top_levels.png",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    plot_full_tree_structure(
        model,
        resolved_figures / "baseline_tree_full_structure.png",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    export_full_tree_dot(
        model,
        feature_names,
        resolved_trees / "baseline_tree_full.dot",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    plot_feature_importance(
        importance, resolved_figures / "top_feature_importance.png", top_n=15
    )
    joblib.dump(model, resolved_trees / "baseline_tree_model.joblib")
    joblib.dump(
        preprocessing_pipeline,
        resolved_shared_output / "preprocessor.joblib",
    )
    np.savez_compressed(
        resolved_shared_output / "split_indices.npz",
        train=train_row_indices,
        test=test_row_indices,
    )

    experiment_identity = build_experiment_identity(
        DATASET_PATH,
        train_row_indices,
        test_row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        target_mapping=TARGET_MAPPING,
    )
    dataset_sha256 = experiment_identity["dataset_sha256"]
    shared_manifest = {
        "dataset_path": str(DATASET_DISPLAY_PATH),
        "dataset_sha256": dataset_sha256,
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "positive_class_name": POSITIVE_CLASS_NAME,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_strategy": SPLIT_STRATEGY,
        "stratified_by": TARGET_COLUMN,
        "train_rows": int(len(split.y_train)),
        "test_rows": int(len(split.y_test)),
        "train_indices_sha256": experiment_identity["train_indices_sha256"],
        "test_indices_sha256": experiment_identity["test_indices_sha256"],
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
        "numeric_processing": "passthrough",
        "preprocessor_fit_partition": "training only",
        "shared_data_entry_point": "src.data.load_and_split_data",
        "shared_preprocessing_entry_point": "src.preprocessing.build_preprocessing_pipeline",
        "target_mapping": {"no": 0, "yes": 1},
        "transformed_feature_names": feature_names.tolist(),
        "experiment_identity": experiment_identity,
    }
    save_json(shared_manifest, resolved_shared_output / "split_manifest.json")

    manifest = {
        "scope": "Hoang baseline Decision Tree and resulting-tree analysis only",
        "dataset_path": str(DATASET_DISPLAY_PATH),
        "dataset_sha256": dataset_sha256,
        "dataset_rows": int(len(split.y_train) + len(split.y_test)),
        "raw_predictors": int(split.X_train.shape[1]),
        "transformed_features": int(len(feature_names)),
        "train_rows": int(len(split.y_train)),
        "test_rows": int(len(split.y_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_indices_sha256": shared_manifest["train_indices_sha256"],
        "test_indices_sha256": shared_manifest["test_indices_sha256"],
        "positive_label": POSITIVE_CLASS_NAME,
        "positive_label_encoded_value": POSITIVE_CLASS,
        "configuration": configuration,
        "experiment_identity": experiment_identity,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_json(manifest, resolved_results / "run_manifest.json")

    render_baseline_report(
        report_path=resolved_report,
        figures_dir=resolved_figures,
        trees_dir=resolved_trees,
        dataset_path=DATASET_DISPLAY_PATH,
        row_count=len(split.y_train) + len(split.y_test),
        raw_feature_count=split.X_train.shape[1],
        transformed_feature_count=len(feature_names),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        configuration=configuration,
        metrics=metrics,
        statistics=statistics,
        importance=importance,
        early_splits=early_splits,
        rules=rules,
        leakage_audit=audit,
    )

    return {
        "metrics": metrics,
        "tree_statistics": statistics,
        "top_features": importance.head(10).to_dict(orient="records"),
        "figures_dir": str(resolved_figures),
        "results_dir": str(resolved_results),
        "trees_dir": str(resolved_trees),
        "shared_output_dir": str(resolved_shared_output),
        "report": str(resolved_report),
    }
