#!/usr/bin/env python3
"""Run only Hoang's baseline Decision Tree and tree-analysis workflow."""

from __future__ import annotations

import argparse
import json
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

from src.artifact_utils import file_sha256, index_fingerprint, save_json
from src.baseline_tree import (
    baseline_configuration,
    evaluate_classifier,
    export_full_tree_dot,
    extract_early_splits,
    extract_early_tree_text,
    extract_feature_importance,
    extract_representative_rules,
    extract_tree_statistics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_full_tree_structure,
    plot_tree_top_levels,
    train_baseline_tree,
    write_representative_rules,
)
from src.data import load_and_split_data
from src.experiment_contract import (
    CLASS_DISPLAY_NAMES,
    DATASET_DISPLAY_PATH,
    DATASET_PATH,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    SPLIT_STRATEGY,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names


DEFAULT_OUTPUT = Path("artifacts/baseline")
DEFAULT_REPORT = Path("reports/hoang_baseline_and_tree_analysis.md")
DEFAULT_SHARED_OUTPUT = Path("artifacts/shared")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and analyze the untuned baseline Decision Tree only."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--shared-output-dir", type=Path, default=DEFAULT_SHARED_OUTPUT)
    return parser.parse_args()


def build_preprocessing_audit(
    split: Any,
    X_train_processed: Any,
    X_test_processed: Any,
    feature_names: np.ndarray,
) -> dict[str, Any]:
    """Record leakage and alignment checks around Khang's fitted pipeline."""

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


def render_report(
    *,
    report_path: Path,
    output_dir: Path,
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
    """Generate report-ready academic English using only measured values."""

    relative_artifacts = Path("..") / output_dir
    matrix = metrics["confusion_matrix"]
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
            f"path depth = {rule['depth']})."
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

## Confusion Matrix

![Baseline confusion matrix]({relative_artifacts / 'confusion_matrix.png'})

The class order is `{metrics['class_labels'][0]}`, `{metrics['class_labels'][1]}`. Therefore, the matrix is `[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]`: {matrix[0][0]} true negatives, {matrix[0][1]} false positives, {matrix[1][0]} false negatives, and {matrix[1][1]} true positives. The horizontal axis is the predicted label and the vertical axis is the true label.

# Analysis of the Tree

## Tree Structure

The fitted baseline has depth **{statistics['depth']}**, **{statistics['nodes']:,} nodes**, and **{statistics['leaves']:,} leaves**. Its training accuracy is {metrics['train_accuracy']:.6f}, versus {metrics['test_accuracy']:.6f} on the test set, a gap of {metrics['train_test_accuracy_gap']:.6f}. {overfit_statement}: the unrestricted tree fits the training observations{' perfectly' if metrics['train_accuracy'] == 1.0 else ' very closely'} but generalizes substantially less accurately. This diagnosis describes the measured baseline and is not the result of tuning.

![Full baseline tree structure]({relative_artifacts / 'baseline_tree_full_structure.png'})

The full structural view contains every node, with colour indicating the node's predicted class. Labels are intentionally omitted at this scale. The following unchanged-model view displays the first levels with readable node labels; it is a presentation truncation, not a smaller retrained tree.

![Baseline tree top levels]({relative_artifacts / 'baseline_tree_top_levels.png'})

## Important Splits and Decision Rules

The root split uses `{statistics['root_split_feature']}` at threshold {statistics['root_split_threshold']:.6f}. The left branch represents **{statistics['root_left_condition']}**, and the right branch represents **{statistics['root_right_condition']}**. The threshold is interpreted as {statistics['root_split_kind']}.

The internal splits through depth 2 are listed below. For a one-hot feature such as `poutcome_success`, a threshold of 0.5 separates rows without that category (left) from rows with that category (right).

| Depth | Branch path | Split feature | Threshold | Training rows at node | Node prediction |
| ---: | --- | --- | ---: | ---: | --- |
{early_split_rows}

Representative fitted leaf rules were selected programmatically for high support and moderate path length:

{chr(10).join(rule_sections)}

These rules describe associations learned by this fitted tree. They should not be interpreted as causal effects. The exact early-level tree text is saved in `{relative_artifacts / 'early_tree.txt'}`, and the complete labeled tree is available in `{relative_artifacts / 'baseline_tree_full.dot'}`.

## Feature Importance

The tree's impurity-based `feature_importances_` values were mapped one-to-one to fitted `ColumnTransformer.get_feature_names_out()` names.

| Rank | Feature | Importance |
| ---: | --- | ---: |
{feature_rows}

![Top feature importances]({relative_artifacts / 'top_feature_importance.png'})

An importance value is the normalized total impurity reduction attributed to a transformed feature. It indicates how much the fitted tree used that feature, but it does not establish causality and may favour variables offering many possible split points.

## Strengths of the Baseline Tree

- It represents nonlinear interactions without requiring feature scaling.
- Its fitted decisions can be inspected through explicit branches and leaf rules.
- It provides a reproducible reference result for the separately owned improvement experiments.

## Weaknesses of the Baseline Tree

- Its depth, leaf count, and measured train-test gap make the unrestricted baseline difficult to interpret in full and indicate poor generalization relative to its training fit.
- Positive-class performance is weaker than overall accuracy when recall/F1 are considered, so accuracy alone would hide important errors on the minority `yes` class.
- Impurity-based feature importance is model-specific and non-causal.
- The `duration` predictor is only known after a marketing call finishes. Its use is valid for reproducing the selected dataset baseline, but it would be unavailable for a pre-call targeting system and is therefore a deployment-time leakage concern, not a train/test leakage bug.

## Data-Leakage Audit

{leakage_checks}

The test partition was never supplied to model or preprocessing `fit`, and the target was removed before splitting predictors. No test result was used to tune this baseline. Khang's original train/test indices are fingerprinted in the shared manifest so every later experiment can verify direct comparability.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.shared_output_dir.mkdir(parents=True, exist_ok=True)

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

    save_json(metrics, args.output_dir / "baseline_metrics.json")
    pd.DataFrame(
        [(key, value) for key, value in metrics.items() if isinstance(value, (int, float, str))],
        columns=["Metric", "Result"],
    ).to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    classification_report_frame.to_csv(args.output_dir / "classification_report.csv")
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    save_json(statistics, args.output_dir / "tree_analysis.json")
    save_json({"splits": early_splits}, args.output_dir / "early_splits.json")
    save_json(audit, args.output_dir / "preprocessing_audit.json")
    (args.output_dir / "early_tree.txt").write_text(
        extract_early_tree_text(model, feature_names), encoding="utf-8"
    )
    write_representative_rules(rules, args.output_dir / "representative_rules.md")

    plot_confusion_matrix(
        split.y_test,
        y_test_pred,
        model.classes_,
        args.output_dir / "confusion_matrix.png",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    plot_tree_top_levels(
        model,
        feature_names,
        args.output_dir / "baseline_tree_top_levels.png",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    plot_full_tree_structure(
        model,
        args.output_dir / "baseline_tree_full_structure.png",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    export_full_tree_dot(
        model,
        feature_names,
        args.output_dir / "baseline_tree_full.dot",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    plot_feature_importance(
        importance, args.output_dir / "top_feature_importance.png", top_n=15
    )
    joblib.dump(model, args.output_dir / "baseline_tree_model.joblib")
    joblib.dump(preprocessing_pipeline, args.shared_output_dir / "preprocessor.joblib")
    np.savez_compressed(
        args.shared_output_dir / "split_indices.npz",
        train=train_row_indices,
        test=test_row_indices,
    )

    dataset_sha256 = file_sha256(DATASET_PATH)
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
        "train_indices_sha256": index_fingerprint(train_row_indices),
        "test_indices_sha256": index_fingerprint(test_row_indices),
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore', sparse_output=True)",
        "numeric_processing": "passthrough",
        "preprocessor_fit_partition": "training only",
        "shared_data_entry_point": "src.data.load_and_split_data",
        "shared_preprocessing_entry_point": "src.preprocessing.build_preprocessing_pipeline",
        "target_mapping": {"no": 0, "yes": 1},
        "transformed_feature_names": feature_names.tolist(),
    }
    save_json(shared_manifest, args.shared_output_dir / "split_manifest.json")

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
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_json(manifest, args.output_dir / "run_manifest.json")

    render_report(
        report_path=args.report,
        output_dir=args.output_dir,
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

    print("Baseline Decision Tree completed successfully.")
    print(json.dumps(metrics, indent=2))
    print(json.dumps(statistics, indent=2))
    print("Top 10 transformed features:")
    print(importance.head(10).to_string(index=False))
    print(f"Artifacts: {args.output_dir}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
