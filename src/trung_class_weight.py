"""Class-weight experiment and final team comparison owned by Trung."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.artifact_utils import index_fingerprint, save_json
from src.baseline_tree import evaluate_classifier, plot_confusion_matrix
from src.data import PROJECT_ROOT, DatasetSplit, load_and_split_data
from src.experiment_contract import (
    CLASS_DISPLAY_NAMES,
    DATASET_DISPLAY_PATH,
    FROZEN_BASELINE_PARAMETERS,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocessing import build_model_pipeline


DEFAULT_FIGURES_OUTPUT = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TREES_OUTPUT = PROJECT_ROOT / "outputs" / "trees"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "trung_class_weight_and_conclusion.md"

CV_FOLDS = 5
SCORING_METRIC = "f1"
CLASS_WEIGHT_CANDIDATES: tuple[Any, ...] = (
    None,
    "balanced",
    {0: 1.0, 1: 1.5},
    {0: 1.0, 1: 2.0},
    {0: 1.0, 1: 3.0},
    {0: 1.0, 1: 4.0},
    {0: 1.0, 1: 5.0},
)


def load_shared_split() -> DatasetSplit:
    """Load the official split shared by every team experiment."""

    return load_and_split_data(test_size=TEST_SIZE, random_state=RANDOM_STATE)


def class_weight_label(class_weight: Any) -> str:
    """Return a stable, readable label for a class-weight candidate."""

    if class_weight is None:
        return "None"
    if class_weight == "balanced":
        return "balanced"
    return f"no=1, yes={float(class_weight[1]):g}"


def build_trung_tree_pipeline(class_weight: Any = None) -> Pipeline:
    """Build a baseline-equivalent tree that changes only ``class_weight``."""

    parameters = dict(FROZEN_BASELINE_PARAMETERS)
    parameters["class_weight"] = class_weight
    return build_model_pipeline(DecisionTreeClassifier(**parameters))


def make_cv_strategy(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """Create the deterministic training-only selection strategy."""

    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def run_class_weight_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    candidates: Sequence[Any] = CLASS_WEIGHT_CANDIDATES,
    cv: StratifiedKFold | None = None,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Evaluate class weights using fold-local preprocessing and no test data."""

    if not candidates:
        raise ValueError("At least one class-weight candidate is required.")
    strategy = cv or make_cv_strategy()
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows: list[dict[str, Any]] = []
    for rank, class_weight in enumerate(candidates):
        scores = cross_validate(
            build_trung_tree_pipeline(class_weight),
            X_train,
            y_train,
            scoring=scoring,
            cv=strategy,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        row: dict[str, Any] = {
            "candidate_order": rank,
            "class_weight_label": class_weight_label(class_weight),
            "class_weight": json.dumps(class_weight, sort_keys=True),
            "cv_folds": strategy.n_splits,
        }
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"mean_cv_{metric}"] = float(values.mean())
            row[f"std_cv_{metric}"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_class_weight(
    validation: pd.DataFrame,
    candidates: Sequence[Any] = CLASS_WEIGHT_CANDIDATES,
) -> tuple[Any, dict[str, Any]]:
    """Select highest CV F1, breaking ties by recall then candidate order."""

    required = {"candidate_order", "mean_cv_f1", "mean_cv_recall"}
    missing = required.difference(validation.columns)
    if missing:
        raise ValueError(f"Validation table is missing columns: {sorted(missing)}")
    ordered = validation.sort_values(
        ["mean_cv_f1", "mean_cv_recall", "candidate_order"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    record = ordered.iloc[0].to_dict()
    candidate_index = int(record["candidate_order"])
    if candidate_index >= len(candidates):
        raise ValueError("Selected candidate index is outside the supplied candidates.")
    return candidates[candidate_index], record


def evaluate_weighted_pipeline(
    fitted_pipeline: Pipeline,
    split: DatasetSplit,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, dict[str, int]]:
    """Evaluate the selected weighted pipeline once on the official test set."""

    metrics, report, predictions = evaluate_classifier(
        fitted_pipeline,
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        positive_label=POSITIVE_CLASS,
        positive_class_name=POSITIVE_CLASS_NAME,
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    model = fitted_pipeline.named_steps["model"]
    complexity = {
        "depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "nodes": int(model.tree_.node_count),
    }
    return metrics, report, predictions, complexity


def _load_metric_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return dict(zip(frame["Metric"], frame["Result"], strict=True))


def load_team_references(
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
) -> list[dict[str, Any]]:
    """Load the frozen, tuned, and validation-selected pruning references."""

    results = Path(results_dir).resolve()
    trees = Path(trees_dir).resolve()
    baseline_metrics = json.loads(
        (results / "baseline_metrics.json").read_text(encoding="utf-8")
    )
    baseline_complexity = json.loads(
        (trees / "tree_analysis.json").read_text(encoding="utf-8")
    )
    hau = _load_metric_csv(results / "hau_tuned_tree_metrics.csv")
    kiet_payload = json.loads(
        (results / "kiet_pruning_metrics.json").read_text(encoding="utf-8")
    )
    selected_kiet = next(
        row
        for row in kiet_payload["comparison"]
        if row["model"] == kiet_payload["selected_model"]
    )
    return [
        {
            "Model": "Hoang Baseline",
            **{key: baseline_metrics[key] for key in _metric_keys()},
            **{key: baseline_complexity[key] for key in _complexity_keys()},
        },
        {
            "Model": "Hau Tuned",
            **{key: float(hau[key]) for key in _metric_keys()},
            "depth": int(float(hau["tree_depth"])),
            "leaves": int(float(hau["number_of_leaves"])),
            "nodes": int(float(hau["node_count"])),
        },
        {
            "Model": f"Kiet Pruned {kiet_payload['selected_criterion'].title()}",
            **{key: selected_kiet[key] for key in _metric_keys()},
            **{key: selected_kiet[key] for key in _complexity_keys()},
        },
    ]


def _metric_keys() -> tuple[str, ...]:
    return ("accuracy", "error_rate", "precision", "recall", "f1_score", "roc_auc")


def _complexity_keys() -> tuple[str, ...]:
    return ("depth", "leaves", "nodes")


def build_team_comparison_table(
    references: Sequence[Mapping[str, Any]],
    weighted_metrics: Mapping[str, Any],
    weighted_complexity: Mapping[str, Any],
) -> pd.DataFrame:
    """Combine the official model from each member into one final table."""

    rows = [dict(reference) for reference in references]
    rows.append(
        {
            "Model": "Trung Class Weight",
            **{key: weighted_metrics[key] for key in _metric_keys()},
            **{key: weighted_complexity[key] for key in _complexity_keys()},
        }
    )
    return pd.DataFrame(rows)


def plot_class_weight_validation(validation: pd.DataFrame, output_path: Path) -> None:
    """Plot CV precision, recall, and F1 for every class-weight candidate."""

    positions = np.arange(len(validation))
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for metric, color in (("precision", "#64748B"), ("recall", "#F59E0B"), ("f1", "#2563EB")):
        axis.plot(
            positions,
            validation[f"mean_cv_{metric}"],
            marker="o",
            linewidth=2,
            label=metric.title(),
            color=color,
        )
    selected_index = int(validation["mean_cv_f1"].idxmax())
    axis.scatter(
        [selected_index],
        [validation.loc[selected_index, "mean_cv_f1"]],
        marker="*",
        s=220,
        color="#DC2626",
        edgecolor="black",
        linewidth=0.5,
        label="Selected by F1",
        zorder=5,
    )
    axis.set_xticks(positions, validation["class_weight_label"], rotation=25, ha="right")
    lower = float(validation[["mean_cv_precision", "mean_cv_recall", "mean_cv_f1"]].min().min())
    upper = float(validation[["mean_cv_precision", "mean_cv_recall", "mean_cv_f1"]].max().max())
    padding = max(0.01, (upper - lower) * 0.35)
    axis.set_ylim(max(0.0, lower - padding), min(1.0, upper + padding))
    axis.set_xlabel("Class weight candidate")
    axis.set_ylabel("Mean cross-validation score")
    axis.set_title("Trung - Class Weight Validation on Training Folds")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_team_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Plot the common held-out metrics for all official team models."""

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    positions = np.arange(len(comparison))
    width = 0.15
    figure, axis = plt.subplots(figsize=(14, 6.2))
    for index, metric in enumerate(metrics):
        axis.bar(
            positions + (index - 2) * width,
            comparison[metric],
            width,
            label=metric.replace("_", " ").title(),
        )
    axis.set_xticks(positions, comparison["Model"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test score")
    axis.set_title("Final Comparison of Team Decision Tree Experiments")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_trung_report(
    *,
    report_path: Path,
    figures_dir: Path,
    selected_label: str,
    selected_validation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    complexity: Mapping[str, Any],
    comparison: pd.DataFrame,
) -> None:
    """Write Trung's method, team comparison, and final conclusion."""

    relative_figures = Path(os.path.relpath(figures_dir, start=report_path.parent))
    rows = "\n".join(
        f"| {row.Model} | {row.accuracy:.6f} | {row.precision:.6f} | "
        f"{row.recall:.6f} | {row.f1_score:.6f} | {row.roc_auc:.6f} | "
        f"{int(row.depth)} | {int(row.leaves):,} | {int(row.nodes):,} |"
        for row in comparison.itertuples(index=False)
    )
    baseline = comparison.iloc[0]
    recall_delta = metrics["recall"] - baseline["recall"]
    f1_delta = metrics["f1_score"] - baseline["f1_score"]
    report = f"""# Improvement Method 3 - Class Weight and Final Conclusion

## Mục tiêu

Phần này do **Trung** phụ trách. Dataset có lớp `yes` thiểu số, vì vậy Accuracy có thể che khuất việc mô hình bỏ sót khách hàng đăng ký tiền gửi. Thí nghiệm thay đổi duy nhất tham số `class_weight` của frozen baseline để tăng chi phí phân loại sai lớp `yes`.

## Thiết kế thí nghiệm

- Giữ nguyên dataset, target, stratified split 80/20 và `random_state=42` của nhóm.
- Preprocessing nằm trong pipeline và được fit riêng trong từng training fold.
- Chọn weight bằng F1 lớp `yes` trên stratified {CV_FOLDS}-fold cross-validation; khi bằng nhau ưu tiên Recall cao hơn.
- Tập test chính thức chỉ được dùng một lần sau khi weight đã được khóa.
- Weight được chọn: **`{selected_label}`**, mean CV F1 **{float(selected_validation['mean_cv_f1']):.6f}**, mean CV Recall **{float(selected_validation['mean_cv_recall']):.6f}**.

![Class-weight validation]({(relative_figures / 'trung_class_weight_validation.png').as_posix()})

## Kết quả mô hình của Trung

| Metric | Result |
| --- | ---: |
| Accuracy | {metrics['accuracy']:.6f} |
| Error Rate | {metrics['error_rate']:.6f} |
| Precision (`yes`) | {metrics['precision']:.6f} |
| Recall (`yes`) | {metrics['recall']:.6f} |
| F1 (`yes`) | {metrics['f1_score']:.6f} |
| ROC-AUC | {metrics['roc_auc']:.6f} |
| Depth | {complexity['depth']} |
| Leaves | {complexity['leaves']:,} |
| Nodes | {complexity['nodes']:,} |

![Weighted confusion matrix]({(relative_figures / 'trung_confusion_matrix.png').as_posix()})

## Comparison of Results

| Model | Accuracy | Precision yes | Recall yes | F1 yes | ROC-AUC | Depth | Leaves | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

![Team model comparison]({(relative_figures / 'team_model_comparison.png').as_posix()})

## Conclusion

So với frozen baseline, mô hình class-weight của Trung làm Recall lớp `yes` thay đổi {recall_delta:+.6f} và F1 thay đổi {f1_delta:+.6f}. Class weighting hướng cây chú ý nhiều hơn tới lỗi false negative nhưng có thể đánh đổi Precision hoặc Accuracy; vì vậy mô hình phù hợp phải được chọn theo chi phí nghiệp vụ, không chỉ theo Accuracy.

Hậu cho thấy giới hạn cấu trúc có thể giảm overfitting, Kiệt cho thấy pruning giảm mạnh kích thước cây, còn Trung tập trung trực tiếp vào mất cân bằng lớp. Bảng trên dùng cùng test partition cho mọi mô hình, nhưng từng cấu hình đều được khóa bằng training-only validation trước khi xem test.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def run_trung_class_weight_workflow(
    *,
    figures_dir: str | Path = DEFAULT_FIGURES_OUTPUT,
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    candidates: Sequence[Any] = CLASS_WEIGHT_CANDIDATES,
    cv_folds: int = CV_FOLDS,
    n_jobs: int | None = -1,
    team_references: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Trung's validation, final evaluation, and team comparison workflow."""

    figures = Path(figures_dir).resolve()
    results = Path(results_dir).resolve()
    trees = Path(trees_dir).resolve()
    report = Path(report_path).resolve()
    for directory in (figures, results, trees, report.parent):
        directory.mkdir(parents=True, exist_ok=True)

    split = load_shared_split()
    validation = run_class_weight_validation(
        split.X_train,
        split.y_train,
        candidates=candidates,
        cv=make_cv_strategy(cv_folds),
        n_jobs=n_jobs,
    )
    validation_path = results / "trung_class_weight_validation.csv"
    validation.to_csv(validation_path, index=False)
    plot_class_weight_validation(validation, figures / "trung_class_weight_validation.png")

    selected_weight, selected_validation = select_best_class_weight(validation, candidates)
    selected_pipeline = build_trung_tree_pipeline(selected_weight)
    selected_pipeline.fit(split.X_train, split.y_train)
    metrics, classification, predictions, complexity = evaluate_weighted_pipeline(
        selected_pipeline, split
    )

    metrics_output = {**metrics, **complexity}
    pd.DataFrame(metrics_output.items(), columns=["Metric", "Result"]).to_csv(
        results / "trung_weighted_tree_metrics.csv", index=False
    )
    classification.to_csv(results / "trung_classification_report.csv")
    plot_confusion_matrix(
        split.y_test,
        predictions,
        selected_pipeline.classes_,
        figures / "trung_confusion_matrix.png",
        class_display_names=CLASS_DISPLAY_NAMES,
    )
    joblib.dump(selected_pipeline, trees / "trung_weighted_pipeline.joblib")

    references = (
        list(team_references)
        if team_references is not None
        else load_team_references(results, trees)
    )
    comparison = build_team_comparison_table(references, metrics, complexity)
    comparison.to_csv(results / "team_model_comparison.csv", index=False)
    plot_team_comparison(comparison, figures / "team_model_comparison.png")

    summary = {
        "scope": "Trung - class imbalance with class_weight and final comparison",
        "dataset_path": str(DATASET_DISPLAY_PATH),
        "selection_metric": SCORING_METRIC,
        "cv_folds": cv_folds,
        "selected_class_weight": selected_weight,
        "selected_class_weight_label": class_weight_label(selected_weight),
        "selected_validation": selected_validation,
        "metrics": metrics,
        "tree_complexity": complexity,
        "train_indices_sha256": index_fingerprint(split.X_train.index.to_numpy()),
        "test_indices_sha256": index_fingerprint(split.X_test.index.to_numpy()),
        "test_partition_usage": "one final evaluation after selection",
    }
    save_json(summary, results / "trung_best_class_weight.json")
    render_trung_report(
        report_path=report,
        figures_dir=figures,
        selected_label=class_weight_label(selected_weight),
        selected_validation=selected_validation,
        metrics=metrics,
        complexity=complexity,
        comparison=comparison,
    )

    return {
        "selected_class_weight": selected_weight,
        "selected_class_weight_label": class_weight_label(selected_weight),
        "metrics": metrics,
        "tree_complexity": complexity,
        "validation_output": str(validation_path),
        "comparison_output": str(results / "team_model_comparison.csv"),
        "report": str(report),
    }
