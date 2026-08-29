"""Shared classifier evaluation and confusion-matrix visualization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _class_names(model: Any, display_names: Sequence[str] | None) -> list[str]:
    """Resolve human-readable names in the fitted classifier's class order."""

    names = (
        [str(label) for label in model.classes_]
        if display_names is None
        else [str(name) for name in display_names]
    )
    if len(names) != len(model.classes_):
        raise ValueError("Class display names do not align with model classes.")
    return names


def evaluate_classifier(
    model: Any,
    X_train: Any,
    y_train: Sequence[Any],
    X_test: Any,
    y_test: Sequence[Any],
    *,
    positive_label: Any,
    positive_class_name: str | None = None,
    class_display_names: Sequence[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Evaluate predictions and positive-class probabilities on held-out data."""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    class_labels = list(model.classes_)
    if positive_label not in class_labels:
        raise ValueError(f"Positive label {positive_label!r} not in {class_labels}.")
    positive_index = class_labels.index(positive_label)
    positive_scores = model.predict_proba(X_test)[:, positive_index]
    display_names = _class_names(model, class_display_names)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    matrix = confusion_matrix(y_test, y_test_pred, labels=class_labels)
    report = classification_report(
        y_test,
        y_test_pred,
        labels=class_labels,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    # Accuracy is a scalar in sklearn's output dictionary. Construct its row
    # explicitly to preserve the text-report semantics and correct support.
    report_accuracy = float(report.pop("accuracy"))
    report_frame = pd.DataFrame(report).transpose()
    report_frame.loc["accuracy"] = {
        "precision": np.nan,
        "recall": np.nan,
        "f1-score": report_accuracy,
        "support": float(len(y_test)),
    }
    report_frame = report_frame.loc[
        [*display_names, "accuracy", "macro avg", "weighted avg"]
    ]
    metrics = {
        "accuracy": float(test_accuracy),
        "error_rate": float(1.0 - test_accuracy),
        "precision": float(
            precision_score(y_test, y_test_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_test_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_test_pred, pos_label=positive_label, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, positive_scores)),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_test_accuracy_gap": float(train_accuracy - test_accuracy),
        "positive_class": positive_class_name or str(positive_label),
        "positive_class_encoded_value": positive_label,
        "class_labels": display_names,
        "zero_division_policy": 0,
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, report_frame, y_test_pred


def plot_confusion_matrix(
    y_test: Sequence[Any],
    y_pred: Sequence[Any],
    class_values: Sequence[Any],
    output_path: str | Path,
    *,
    class_display_names: Sequence[str] | None = None,
    title: str = "Decision Tree - Confusion Matrix",
) -> None:
    """Save a report-ready confusion matrix with correct axis semantics."""

    matrix = confusion_matrix(y_test, y_pred, labels=class_values)
    display_names = (
        [str(label) for label in class_values]
        if class_display_names is None
        else list(class_display_names)
    )
    if len(display_names) != len(class_values):
        raise ValueError("Class display names do not align with class values.")
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    display = ConfusionMatrixDisplay(matrix, display_labels=display_names)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(title, pad=14)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
