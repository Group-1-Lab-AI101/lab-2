"""Cross-validated Decision Tree hyperparameter experiments owned by Hau.

This module reuses the project's frozen data/split contract and shared model
pipeline.  Hyperparameters are selected exclusively from training-fold F1
scores; the official test partition is evaluated only after ``GridSearchCV``
has selected and refitted its best pipeline on the complete training set.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.artifact_utils import index_fingerprint, save_json
from src.data import DatasetSplit, PROJECT_ROOT, load_and_split_data
from src.experiment_contract import (
    DATASET_DISPLAY_PATH,
    DATASET_PATH,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    SPLIT_STRATEGY,
    TEST_SIZE,
)
from src.preprocessing import build_model_pipeline


DEFAULT_FIGURES_OUTPUT = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TREES_OUTPUT = PROJECT_ROOT / "outputs" / "trees"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "hau_improvement_methods.md"

SCORING_METRIC = "f1"
CV_FOLDS = 5
TUNED_PARAMETERS = (
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
)

SINGLE_PARAMETER_CANDIDATES: Mapping[str, tuple[Any, ...]] = {
    "max_depth": (3, 5, 7, 10, 15, 20, None),
    "min_samples_split": (2, 5, 10, 20, 50, 100),
    "min_samples_leaf": (1, 2, 5, 10, 20, 50),
}

COMBINED_SEARCH_SPACE: Mapping[str, tuple[Any, ...]] = {
    "max_depth": (5, 7, 10, 15, 20, None),
    "min_samples_split": (2, 5, 10, 20, 50, 100),
    "min_samples_leaf": (1, 2, 5, 10, 20, 50),
}

VALIDATION_OUTPUT_NAMES = {
    "max_depth": "hau_max_depth_validation.csv",
    "min_samples_split": "hau_min_samples_split_validation.csv",
    "min_samples_leaf": "hau_min_samples_leaf_validation.csv",
}

VALIDATION_FIGURE_NAMES = {
    "max_depth": "hau_max_depth_validation.png",
    "min_samples_split": "hau_min_samples_split_validation.png",
    "min_samples_leaf": "hau_min_samples_leaf_validation.png",
}


def load_shared_split() -> DatasetSplit:
    """Load the one official stratified train/test partition for Hau's work."""

    return load_and_split_data(
        DATASET_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def build_hau_tree_pipeline() -> Pipeline:
    """Build an unfitted Decision Tree inside Khang's shared preprocessing."""

    estimator = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return build_model_pipeline(estimator)


def estimator_step_name(pipeline: Pipeline) -> str:
    """Discover the Decision Tree step instead of assuming a pipeline prefix."""

    matches = [
        name
        for name, estimator in pipeline.named_steps.items()
        if isinstance(estimator, DecisionTreeClassifier)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one DecisionTreeClassifier pipeline step; "
            f"found {matches}."
        )
    return matches[0]


def make_cv_strategy(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """Create the deterministic stratified CV strategy used for selection."""

    if n_splits < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_combined_parameter_grid(
    pipeline: Pipeline,
    search_space: Mapping[str, Sequence[Any]] = COMBINED_SEARCH_SPACE,
) -> dict[str, list[Any]]:
    """Prefix Hau's three owned parameters using the inspected model step."""

    supplied = set(search_space)
    expected = set(TUNED_PARAMETERS)
    if supplied != expected:
        raise ValueError(
            "Hau's search space must contain exactly "
            f"{sorted(expected)}; got {sorted(supplied)}."
        )
    step_name = estimator_step_name(pipeline)
    return {
        f"{step_name}__{parameter}": list(search_space[parameter])
        for parameter in TUNED_PARAMETERS
    }


def run_single_parameter_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    parameter: str,
    candidates: Sequence[Any],
    cv: StratifiedKFold | None = None,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Measure one owned parameter with fold-local preprocessing and F1."""

    if parameter not in TUNED_PARAMETERS:
        raise ValueError(
            f"Unsupported Hau parameter {parameter!r}; expected {TUNED_PARAMETERS}."
        )
    if not candidates:
        raise ValueError("At least one validation candidate is required.")

    pipeline = build_hau_tree_pipeline()
    full_parameter_name = f"{estimator_step_name(pipeline)}__{parameter}"
    _, validation_scores = validation_curve(
        pipeline,
        X_train,
        y_train,
        param_name=full_parameter_name,
        param_range=list(candidates),
        cv=cv or make_cv_strategy(),
        scoring=SCORING_METRIC,
        n_jobs=n_jobs,
        error_score="raise",
    )
    return pd.DataFrame(
        {
            "parameter": parameter,
            "parameter_value": list(candidates),
            "mean_cv_score": validation_scores.mean(axis=1),
            "std_cv_score": validation_scores.std(axis=1),
            "scoring_metric": SCORING_METRIC,
            "cv_folds": validation_scores.shape[1],
        }
    )


def run_combined_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    search_space: Mapping[str, Sequence[Any]] = COMBINED_SEARCH_SPACE,
    cv: StratifiedKFold | None = None,
    n_jobs: int | None = -1,
) -> GridSearchCV:
    """Fit the combined training-only search and refit its selected pipeline."""

    pipeline = build_hau_tree_pipeline()
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=build_combined_parameter_grid(pipeline, search_space),
        scoring=SCORING_METRIC,
        cv=cv or make_cv_strategy(),
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def search_results_frame(search: GridSearchCV) -> pd.DataFrame:
    """Return every combined-search candidate in a readable tabular schema."""

    frame = pd.DataFrame(search.cv_results_)
    step_name = estimator_step_name(search.estimator)
    parameter_columns = [f"param_{step_name}__{name}" for name in TUNED_PARAMETERS]
    score_columns = [
        column
        for column in frame.columns
        if column.startswith("split") and column.endswith("_test_score")
    ]
    other_columns = [
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    selected = frame.loc[:, parameter_columns + score_columns + other_columns].copy()
    selected = selected.rename(
        columns={
            f"param_{step_name}__{name}": name for name in TUNED_PARAMETERS
        }
    )
    return selected.sort_values(
        ["rank_test_score", *TUNED_PARAMETERS],
        na_position="last",
        ignore_index=True,
    )


def selected_parameters(search: GridSearchCV) -> dict[str, Any]:
    """Remove the discovered pipeline prefix from the best parameters."""

    step_name = estimator_step_name(search.estimator)
    prefix = f"{step_name}__"
    parameters = {
        name.removeprefix(prefix): value
        for name, value in search.best_params_.items()
    }
    if set(parameters) != set(TUNED_PARAMETERS):
        raise RuntimeError(f"Unexpected selected parameters: {parameters}")
    return {name: parameters[name] for name in TUNED_PARAMETERS}


def extract_fitted_tree(fitted_pipeline: Pipeline) -> DecisionTreeClassifier:
    """Extract the already-fitted selected tree without training another model."""

    tree = fitted_pipeline.named_steps[estimator_step_name(fitted_pipeline)]
    if not hasattr(tree, "tree_"):
        raise ValueError("The selected Decision Tree pipeline has not been fitted.")
    return tree


def evaluate_selected_pipeline(
    fitted_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Evaluate the selected pipeline once on the official held-out test set."""

    y_train_pred = fitted_pipeline.predict(X_train)
    y_test_pred = fitted_pipeline.predict(X_test)
    classes = list(fitted_pipeline.classes_)
    if POSITIVE_CLASS not in classes:
        raise ValueError(f"Positive class {POSITIVE_CLASS!r} not found in {classes}.")
    positive_index = classes.index(POSITIVE_CLASS)
    positive_scores = fitted_pipeline.predict_proba(X_test)[:, positive_index]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    metrics = {
        "accuracy": float(test_accuracy),
        "error_rate": float(1.0 - test_accuracy),
        "precision": float(
            precision_score(
                y_test,
                y_test_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_test_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                y_test_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_test, positive_scores)),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_test_accuracy_gap": float(train_accuracy - test_accuracy),
        "positive_class": POSITIVE_CLASS_NAME,
        "positive_class_encoded_value": POSITIVE_CLASS,
        "zero_division_policy": 0,
    }
    fitted_tree = extract_fitted_tree(fitted_pipeline)
    complexity = {
        "depth": int(fitted_tree.get_depth()),
        "leaves": int(fitted_tree.get_n_leaves()),
        "nodes": int(fitted_tree.tree_.node_count),
    }
    return metrics, complexity


def load_baseline_reference(
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read Hoang's generated frozen metrics and complexity artifacts."""

    metrics_path = Path(results_dir).expanduser().resolve() / "baseline_metrics.json"
    statistics_path = Path(trees_dir).expanduser().resolve() / "tree_analysis.json"
    if not metrics_path.is_file() or not statistics_path.is_file():
        raise FileNotFoundError(
            "Frozen baseline artifacts are required before Hau's comparison: "
            f"{metrics_path}, {statistics_path}"
        )
    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        json.loads(statistics_path.read_text(encoding="utf-8")),
    )


def build_comparison_table(
    baseline_metrics: Mapping[str, Any],
    baseline_complexity: Mapping[str, Any],
    tuned_metrics: Mapping[str, Any],
    tuned_complexity: Mapping[str, Any],
) -> pd.DataFrame:
    """Build a direct baseline-versus-tuned metric and complexity comparison."""

    rows = (
        ("Accuracy", "accuracy", baseline_metrics, tuned_metrics),
        ("Error Rate", "error_rate", baseline_metrics, tuned_metrics),
        ("Precision", "precision", baseline_metrics, tuned_metrics),
        ("Recall", "recall", baseline_metrics, tuned_metrics),
        ("F1", "f1_score", baseline_metrics, tuned_metrics),
        ("ROC-AUC", "roc_auc", baseline_metrics, tuned_metrics),
        ("Tree Depth", "depth", baseline_complexity, tuned_complexity),
        ("Number of Leaves", "leaves", baseline_complexity, tuned_complexity),
    )
    records = []
    for display_name, key, baseline_source, tuned_source in rows:
        baseline_value = float(baseline_source[key])
        tuned_value = float(tuned_source[key])
        records.append(
            {
                "Metric": display_name,
                "Baseline": baseline_value,
                "Hau Tuned": tuned_value,
                "Difference": tuned_value - baseline_value,
            }
        )
    return pd.DataFrame(records)


def plot_validation_results(
    results: pd.DataFrame,
    parameter: str,
    output_path: str | Path,
) -> Path:
    """Save one categorical-axis validation curve with CV standard deviations."""

    labels = ["None" if pd.isna(value) else str(value) for value in results["parameter_value"]]
    positions = np.arange(len(results))
    figure, axis = plt.subplots(figsize=(8.5, 5.3))
    axis.errorbar(
        positions,
        results["mean_cv_score"],
        yerr=results["std_cv_score"],
        marker="o",
        linewidth=1.8,
        capsize=4,
        color="#2563EB",
    )
    axis.set_xticks(positions, labels=labels)
    axis.set_xlabel(parameter)
    fold_count = int(results["cv_folds"].iloc[0])
    axis.set_ylabel(f"Mean {fold_count}-fold validation F1")
    axis.set_title(f"Hau Decision Tree Validation: {parameter}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return resolved


def plot_baseline_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot predictive metrics and tree complexity on separate readable scales."""

    metric_rows = comparison.iloc[:6]
    complexity_rows = comparison.iloc[6:]
    figure, (metric_axis, complexity_axis) = plt.subplots(1, 2, figsize=(14, 5.8))

    metric_positions = np.arange(len(metric_rows))
    width = 0.36
    metric_axis.bar(
        metric_positions - width / 2,
        metric_rows["Baseline"],
        width,
        label="Hoang baseline",
        color="#64748B",
    )
    metric_axis.bar(
        metric_positions + width / 2,
        metric_rows["Hau Tuned"],
        width,
        label="Hau tuned",
        color="#2563EB",
    )
    metric_axis.set_xticks(metric_positions, labels=metric_rows["Metric"], rotation=25)
    metric_axis.set_ylim(0, 1)
    metric_axis.set_ylabel("Score")
    metric_axis.set_title("Held-out Predictive Metrics")
    metric_axis.grid(axis="y", alpha=0.22)
    metric_axis.legend()

    complexity_positions = np.arange(len(complexity_rows))
    complexity_axis.bar(
        complexity_positions - width / 2,
        complexity_rows["Baseline"],
        width,
        label="Hoang baseline",
        color="#64748B",
    )
    complexity_axis.bar(
        complexity_positions + width / 2,
        complexity_rows["Hau Tuned"],
        width,
        label="Hau tuned",
        color="#2563EB",
    )
    complexity_axis.set_xticks(complexity_positions, labels=complexity_rows["Metric"])
    complexity_axis.set_yscale("log")
    complexity_axis.set_ylabel("Count (log scale)")
    complexity_axis.set_title("Fitted Tree Complexity")
    complexity_axis.grid(axis="y", alpha=0.22)
    complexity_axis.legend()

    figure.suptitle("Hoang Frozen Baseline vs Hau Tuned Decision Tree")
    figure.tight_layout()
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return resolved


def _change_sentence(metric: str, difference: float) -> str:
    """Describe the direction of one measured tuned-minus-baseline delta."""

    if np.isclose(difference, 0.0):
        return f"{metric} was unchanged"
    direction = "increased" if difference > 0 else "decreased"
    return f"{metric} {direction} by {abs(difference):.6f}"


def render_hau_report(
    *,
    report_path: str | Path,
    figures_dir: str | Path,
    best_parameters: Mapping[str, Any],
    best_cv_score: float,
    cv_folds: int,
    single_parameter_candidates: Mapping[str, Sequence[Any]],
    combined_search_space: Mapping[str, Sequence[Any]],
    comparison: pd.DataFrame,
    tuned_complexity: Mapping[str, int],
) -> Path:
    """Render Hau's method explanation and actual generated comparison values."""

    resolved_report = Path(report_path).expanduser().resolve()
    resolved_figures = Path(figures_dir).expanduser().resolve()
    relative_figures = Path(
        os.path.relpath(resolved_figures, start=resolved_report.parent)
    )
    comparison_rows = []
    for _, row in comparison.iterrows():
        is_complexity = row["Metric"] in {"Tree Depth", "Number of Leaves"}
        if is_complexity:
            comparison_rows.append(
                f"| {row['Metric']} | {row['Baseline']:.0f} | "
                f"{row['Hau Tuned']:.0f} | {row['Difference']:+.0f} |"
            )
        else:
            comparison_rows.append(
                f"| {row['Metric']} | {row['Baseline']:.6f} | "
                f"{row['Hau Tuned']:.6f} | {row['Difference']:+.6f} |"
            )

    differences = comparison.set_index("Metric")["Difference"]
    metric_summary = "; ".join(
        _change_sentence(metric, float(differences[metric]))
        for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
    )
    error_difference = float(differences["Error Rate"])
    error_summary = _change_sentence("Error rate", error_difference)
    complexity_summary = (
        f"Tree depth changed from {comparison.iloc[6]['Baseline']:.0f} to "
        f"{comparison.iloc[6]['Hau Tuned']:.0f}, and leaves changed from "
        f"{comparison.iloc[7]['Baseline']:.0f} to "
        f"{comparison.iloc[7]['Hau Tuned']:.0f}."
    )
    parameter_text = ", ".join(
        f"`{name}={best_parameters[name]!r}`" for name in TUNED_PARAMETERS
    )
    single_range_rows = "\n".join(
        f"- `{name}`: `{list(single_parameter_candidates[name])}`"
        for name in TUNED_PARAMETERS
    )
    combined_range_rows = "\n".join(
        f"- `{name}`: `{list(combined_search_space[name])}`"
        for name in TUNED_PARAMETERS
    )

    report = f"""# Improvement Method 1 — Hyperparameter Tuning

## Experiment Contract and Selection Method

Hau's experiment reuses `data/bank-full.csv`, target `y` with `yes=1`, and the official stratified 80%/20% split with `random_state=42`. It uses `src.data.load_and_split_data()` and `src.preprocessing.build_model_pipeline()`, so one-hot preprocessing is fitted independently inside each training fold. The held-out test partition is not supplied to validation or model selection.

Candidate configurations are selected by positive-class F1 using shuffled stratified {cv_folds}-fold cross-validation on the training partition. The selected pipeline is then refitted on all training rows and evaluated once on the official test partition. The combined search selected {parameter_text}, with mean validation F1 **{best_cv_score:.6f}**.

The single-parameter validation ranges were:

{single_range_rows}

The combined search ranges were:

{combined_range_rows}

## Baseline Problem

Hoang's frozen baseline is an unrestricted Decision Tree. It can continue creating branches that fit noise or rare training patterns, increasing variance and causing overfitting. Hau's experiment does not modify or redefine that baseline.

## `max_depth`

`max_depth` limits how many split levels the tree can grow. A smaller depth reduces model complexity and variance and can improve generalization; a value that is too small can remove useful interactions and underfit.

![max_depth validation curve]({relative_figures / VALIDATION_FIGURE_NAMES['max_depth']})

## `min_samples_split`

`min_samples_split` is the minimum number of observations required before an internal node may split. Increasing it prevents very small nodes from creating highly specific branches and can reduce overfitting.

![min_samples_split validation curve]({relative_figures / VALIDATION_FIGURE_NAMES['min_samples_split']})

## `min_samples_leaf`

`min_samples_leaf` is the minimum number of observations allowed in every leaf. Increasing it prevents leaves supported by only a few observations, makes predictions less sensitive to individual training cases, and smooths the learned decision boundary.

![min_samples_leaf validation curve]({relative_figures / VALIDATION_FIGURE_NAMES['min_samples_leaf']})

## Cross-validation

Cross-validation estimates how candidate configurations generalize without consuming the official test set for model selection. Because preprocessing is part of the model pipeline, each fold learns its encoder only from that fold's training rows and transforms its validation rows without refitting.

## Frozen Baseline vs Hau Tuned Tree

| Metric | Baseline | Hau Tuned | Difference |
| --- | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

![Baseline versus tuned comparison]({relative_figures / 'hau_baseline_vs_tuned.png'})

## Final Conclusion

On the untouched test set, {error_summary}. {metric_summary}. {complexity_summary} The selected tree contains {tuned_complexity['nodes']:,} total nodes.

These results must be read as a trade-off rather than an accuracy-only claim. F1 was the training-only selection criterion because `yes` is the minority class; recall, precision, and probability-based ROC-AUC show whether any accuracy change also helps subscription detection. A metric that decreased is reported directly rather than being hidden by gains elsewhere. Tuning can improve generalization by constraining high-variance branches, but the measured table above—not the method alone—determines whether this selected model is preferable for the project's goals.
"""
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    resolved_report.write_text(report, encoding="utf-8")
    return resolved_report


def run_hau_hyperparameter_workflow(
    *,
    figures_dir: str | Path = DEFAULT_FIGURES_OUTPUT,
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    baseline_metrics: Mapping[str, Any] | None = None,
    baseline_complexity: Mapping[str, Any] | None = None,
    single_parameter_candidates: Mapping[
        str, Sequence[Any]
    ] = SINGLE_PARAMETER_CANDIDATES,
    combined_search_space: Mapping[str, Sequence[Any]] = COMBINED_SEARCH_SPACE,
    cv_folds: int = CV_FOLDS,
    n_jobs: int | None = -1,
) -> dict[str, Any]:
    """Run, persist, compare, and document Hau's complete owned experiment."""

    if set(single_parameter_candidates) != set(TUNED_PARAMETERS):
        raise ValueError(
            "Single-parameter experiments must cover exactly Hau's three parameters."
        )
    resolved_figures = Path(figures_dir).expanduser().resolve()
    resolved_results = Path(results_dir).expanduser().resolve()
    resolved_trees = Path(trees_dir).expanduser().resolve()
    resolved_report = Path(report_path).expanduser().resolve()
    resolved_figures.mkdir(parents=True, exist_ok=True)
    resolved_results.mkdir(parents=True, exist_ok=True)

    if baseline_metrics is None or baseline_complexity is None:
        loaded_metrics, loaded_complexity = load_baseline_reference(
            resolved_results,
            resolved_trees,
        )
        baseline_metrics = loaded_metrics
        baseline_complexity = loaded_complexity

    split = load_shared_split()
    cv = make_cv_strategy(cv_folds)
    validation_outputs: dict[str, str] = {}
    for parameter in TUNED_PARAMETERS:
        validation = run_single_parameter_validation(
            split.X_train,
            split.y_train,
            parameter=parameter,
            candidates=single_parameter_candidates[parameter],
            cv=cv,
            n_jobs=n_jobs,
        )
        csv_path = resolved_results / VALIDATION_OUTPUT_NAMES[parameter]
        validation.to_csv(csv_path, index=False)
        plot_validation_results(
            validation,
            parameter,
            resolved_figures / VALIDATION_FIGURE_NAMES[parameter],
        )
        validation_outputs[parameter] = str(csv_path)

    search = run_combined_search(
        split.X_train,
        split.y_train,
        search_space=combined_search_space,
        cv=cv,
        n_jobs=n_jobs,
    )
    all_search_results = search_results_frame(search)
    search_results_path = resolved_results / "hau_hyperparameter_search.csv"
    all_search_results.to_csv(search_results_path, index=False)

    best_parameters = selected_parameters(search)
    best_summary = {
        "best_parameters": best_parameters,
        "best_cv_score": float(search.best_score_),
        "scoring_metric": SCORING_METRIC,
        "cv_strategy": (
            f"StratifiedKFold(n_splits={cv_folds}, shuffle=True, "
            f"random_state={RANDOM_STATE})"
        ),
        "candidate_combinations": int(len(all_search_results)),
        "dataset_path": str(DATASET_DISPLAY_PATH),
        "split_strategy": SPLIT_STRATEGY,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(split.y_train)),
        "test_rows": int(len(split.y_test)),
        "train_indices_sha256": index_fingerprint(split.X_train.index.to_numpy()),
        "test_indices_sha256": index_fingerprint(split.X_test.index.to_numpy()),
        "preprocessing": "src.preprocessing.build_model_pipeline",
        "preprocessing_cv_fit_scope": "each training fold only",
        "test_partition_usage": "one final evaluation after selection",
        "estimator_step": estimator_step_name(search.best_estimator_),
    }
    best_parameters_path = resolved_results / "hau_best_parameters.json"
    save_json(best_summary, best_parameters_path)

    tuned_metrics, tuned_complexity = evaluate_selected_pipeline(
        search.best_estimator_,
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
    )
    tuned_output = {
        **tuned_metrics,
        "tree_depth": tuned_complexity["depth"],
        "number_of_leaves": tuned_complexity["leaves"],
        "node_count": tuned_complexity["nodes"],
    }
    tuned_metrics_path = resolved_results / "hau_tuned_tree_metrics.csv"
    pd.DataFrame(
        [(name, value) for name, value in tuned_output.items()],
        columns=["Metric", "Result"],
    ).to_csv(tuned_metrics_path, index=False)

    comparison = build_comparison_table(
        baseline_metrics,
        baseline_complexity,
        tuned_metrics,
        tuned_complexity,
    )
    comparison_path = resolved_results / "hau_baseline_vs_tuned.csv"
    comparison.to_csv(comparison_path, index=False)
    comparison_figure = plot_baseline_comparison(
        comparison,
        resolved_figures / "hau_baseline_vs_tuned.png",
    )
    render_hau_report(
        report_path=resolved_report,
        figures_dir=resolved_figures,
        best_parameters=best_parameters,
        best_cv_score=float(search.best_score_),
        cv_folds=cv_folds,
        single_parameter_candidates=single_parameter_candidates,
        combined_search_space=combined_search_space,
        comparison=comparison,
        tuned_complexity=tuned_complexity,
    )

    return {
        "best_parameters": best_parameters,
        "best_cv_score": float(search.best_score_),
        "scoring_metric": SCORING_METRIC,
        "cv_folds": cv_folds,
        "metrics": tuned_metrics,
        "tree_complexity": tuned_complexity,
        "comparison": comparison.to_dict(orient="records"),
        "validation_outputs": validation_outputs,
        "search_results": str(search_results_path),
        "best_parameters_output": str(best_parameters_path),
        "tuned_metrics_output": str(tuned_metrics_path),
        "comparison_output": str(comparison_path),
        "comparison_figure": str(comparison_figure),
        "report": str(resolved_report),
    }
