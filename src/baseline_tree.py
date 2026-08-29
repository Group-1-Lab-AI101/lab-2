"""Training, visualization, and structural interpretation of the baseline tree."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree

from src.experiment_contract import FROZEN_BASELINE_PARAMETERS


def _class_names(
    model: DecisionTreeClassifier,
    display_names: Sequence[str] | None,
) -> list[str]:
    """Resolve human-readable names in the estimator's class order."""

    names = (
        [str(label) for label in model.classes_]
        if display_names is None
        else [str(name) for name in display_names]
    )
    if len(names) != len(model.classes_):
        raise ValueError("Class display names do not align with model classes.")
    return names


def train_baseline_tree(
    X_train: np.ndarray,
    y_train: Sequence[Any],
) -> DecisionTreeClassifier:
    """Fit the explicit, frozen, untuned baseline configuration."""

    model = DecisionTreeClassifier(**dict(FROZEN_BASELINE_PARAMETERS))
    model.fit(X_train, y_train)
    return model


def baseline_configuration(model: DecisionTreeClassifier) -> dict[str, Any]:
    """Record the exact sklearn configuration of the fitted baseline."""

    params = model.get_params(deep=False)
    configuration = {
        key: params.get(key) for key in FROZEN_BASELINE_PARAMETERS if key in params
    }
    expected = dict(FROZEN_BASELINE_PARAMETERS)
    if configuration != expected:
        raise AssertionError(
            f"Baseline configuration changed: expected {expected}, got {configuration}"
        )
    return configuration


def plot_tree_top_levels(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    output_path: str | Path,
    *,
    max_depth: int = 3,
    class_display_names: Sequence[str] | None = None,
) -> None:
    """Plot early levels of the already-fitted tree; this does not retrain it."""

    figure, axis = plt.subplots(figsize=(28, 14))
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=_class_names(model, class_display_names),
        filled=True,
        rounded=True,
        proportion=True,
        precision=3,
        max_depth=max_depth,
        fontsize=8,
        ax=axis,
    )
    axis.set_title(f"Baseline Decision Tree - Top Levels (depth 0-{max_depth})", fontsize=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_full_tree_structure(
    model: DecisionTreeClassifier,
    output_path: str | Path,
    *,
    class_display_names: Sequence[str] | None = None,
) -> None:
    """Plot every node as a scalable structural overview without unreadable labels."""

    tree = model.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    node_depth = np.zeros(tree.node_count, dtype=int)
    node_x = np.zeros(tree.node_count, dtype=float)
    leaf_counter = 0

    def assign_positions(node_id: int, depth: int) -> float:
        nonlocal leaf_counter
        node_depth[node_id] = depth
        left = children_left[node_id]
        right = children_right[node_id]
        if left == right:
            node_x[node_id] = leaf_counter
            leaf_counter += 1
        else:
            left_x = assign_positions(left, depth + 1)
            right_x = assign_positions(right, depth + 1)
            node_x[node_id] = (left_x + right_x) / 2.0
        return node_x[node_id]

    assign_positions(0, 0)
    segments: list[list[tuple[float, float]]] = []
    for node_id in range(tree.node_count):
        for child_id in (children_left[node_id], children_right[node_id]):
            if child_id != -1:
                segments.append(
                    [
                        (node_x[node_id], -node_depth[node_id]),
                        (node_x[child_id], -node_depth[child_id]),
                    ]
                )

    predicted_class_index = np.argmax(tree.value[:, 0, :], axis=1)
    figure, axis = plt.subplots(figsize=(24, 12))
    axis.add_collection(LineCollection(segments, colors="#808080", linewidths=0.22, alpha=0.55))
    scatter = axis.scatter(
        node_x,
        -node_depth,
        c=predicted_class_index,
        cmap="coolwarm",
        s=np.maximum(1.0, 14.0 * tree.weighted_n_node_samples / tree.weighted_n_node_samples[0]),
        alpha=0.85,
        linewidths=0,
    )
    handles, _ = scatter.legend_elements()
    axis.legend(handles, _class_names(model, class_display_names), title="Predicted class")
    axis.set_title(
        f"Baseline Decision Tree - Full Structure ({tree.node_count:,} nodes, "
        f"{model.get_n_leaves():,} leaves)"
    )
    axis.set_xlabel("Leaf-order position (structural layout; node labels omitted)")
    axis.set_ylabel("Tree depth")
    depths = np.arange(0, model.get_depth() + 1, max(1, model.get_depth() // 10))
    axis.set_yticks(-depths, labels=[str(depth) for depth in depths])
    axis.autoscale()
    axis.margins(x=0.01, y=0.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def export_full_tree_dot(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    output_path: str | Path,
    *,
    class_display_names: Sequence[str] | None = None,
) -> None:
    """Export the full labeled tree in Graphviz DOT format for detailed inspection."""

    export_graphviz(
        model,
        out_file=str(output_path),
        feature_names=list(feature_names),
        class_names=_class_names(model, class_display_names),
        filled=True,
        rounded=True,
        special_characters=False,
        precision=3,
    )


def extract_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Map every fitted importance to the corresponding transformed feature."""

    if len(feature_names) != len(model.feature_importances_):
        raise ValueError("Feature names and model importances have different lengths.")
    table = pd.DataFrame(
        {"Feature": list(feature_names), "Importance": model.feature_importances_}
    ).sort_values(["Importance", "Feature"], ascending=[False, True], ignore_index=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))
    return table


def plot_feature_importance(
    importance_table: pd.DataFrame,
    output_path: str | Path,
    *,
    top_n: int = 15,
) -> None:
    """Save a horizontal chart of the most important transformed features."""

    top = importance_table.head(top_n).iloc[::-1]
    figure, axis = plt.subplots(figsize=(10, 7.5))
    axis.barh(top["Feature"], top["Importance"], color="#3572A5")
    axis.set_title(f"Baseline Decision Tree - Top {min(top_n, len(top))} Feature Importances")
    axis.set_xlabel("Mean decrease in impurity importance")
    axis.set_ylabel("Transformed feature")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _condition(feature: str, threshold: float, go_left: bool) -> str:
    """Render numeric and one-hot split conditions accurately."""

    if "_" in feature and np.isclose(threshold, 0.5):
        raw_feature, category = feature.split("_", 1)
        return (
            f"{raw_feature} is not {category!r}"
            if go_left
            else f"{raw_feature} is {category!r}"
        )
    operator = "<=" if go_left else ">"
    return f"{feature} {operator} {threshold:.3f}"


def extract_tree_statistics(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Obtain structural statistics and root split information."""

    tree = model.tree_
    root_feature_index = int(tree.feature[0])
    root_feature = str(feature_names[root_feature_index])
    root_threshold = float(tree.threshold[0])
    root_kind = (
        "one-hot categorical indicator"
        if "_" in root_feature and np.isclose(root_threshold, 0.5)
        else "numeric"
    )
    return {
        "depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "nodes": int(tree.node_count),
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "train_test_accuracy_gap": metrics["train_test_accuracy_gap"],
        "root_split_feature": root_feature,
        "root_split_threshold": root_threshold,
        "root_split_kind": root_kind,
        "root_left_condition": _condition(root_feature, root_threshold, True),
        "root_right_condition": _condition(root_feature, root_threshold, False),
        "root_samples": int(tree.n_node_samples[0]),
    }


def extract_early_tree_text(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    *,
    max_depth: int = 3,
) -> str:
    """Extract a concise text representation of the first tree levels."""

    return export_text(
        model,
        feature_names=list(feature_names),
        max_depth=max_depth,
        decimals=3,
        show_weights=True,
    )


def extract_early_splits(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    *,
    max_depth: int = 2,
    class_display_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return interpretable metadata for all internal nodes through ``max_depth``."""

    tree = model.tree_
    splits: list[dict[str, Any]] = []
    display_names = _class_names(model, class_display_names)

    def walk(node_id: int, depth: int, branch_path: str) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right or depth > max_depth:
            return
        feature = str(feature_names[tree.feature[node_id]])
        threshold = float(tree.threshold[node_id])
        predicted_index = int(np.argmax(tree.value[node_id][0]))
        splits.append(
            {
                "depth": depth,
                "branch_path": branch_path,
                "feature": feature,
                "threshold": threshold,
                "left_condition": _condition(feature, threshold, True),
                "right_condition": _condition(feature, threshold, False),
                "training_samples": int(tree.n_node_samples[node_id]),
                "predicted_class": display_names[predicted_index],
            }
        )
        walk(left, depth + 1, f"{branch_path}L")
        walk(right, depth + 1, f"{branch_path}R")

    walk(0, 0, "root")
    return splits


def extract_representative_rules(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    *,
    max_rules_per_class: int = 2,
    preferred_max_depth: int = 10,
    class_display_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select high-support, reasonably short leaf paths for each predicted class."""

    tree = model.tree_
    candidates: list[dict[str, Any]] = []
    display_names = _class_names(model, class_display_names)

    def walk(node_id: int, conditions: list[str]) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            counts = tree.value[node_id][0]
            predicted_index = int(np.argmax(counts))
            candidates.append(
                {
                    "leaf_id": int(node_id),
                    "predicted_class": display_names[predicted_index],
                    "predicted_class_value": model.classes_[predicted_index].item()
                    if isinstance(model.classes_[predicted_index], np.generic)
                    else model.classes_[predicted_index],
                    "sample_count": int(tree.n_node_samples[node_id]),
                    "weighted_sample_count": float(tree.weighted_n_node_samples[node_id]),
                    "purity": float(counts[predicted_index] / counts.sum()),
                    "depth": len(conditions),
                    "conditions": conditions,
                }
            )
            return
        feature = str(feature_names[tree.feature[node_id]])
        threshold = float(tree.threshold[node_id])
        walk(left, conditions + [_condition(feature, threshold, True)])
        walk(right, conditions + [_condition(feature, threshold, False)])

    walk(0, [])
    selected: list[dict[str, Any]] = []
    for class_label in display_names:
        class_candidates = [
            candidate
            for candidate in candidates
            if candidate["predicted_class"] == class_label
        ]
        short = [
            candidate
            for candidate in class_candidates
            if candidate["depth"] <= preferred_max_depth
        ]
        pool = short or class_candidates
        pool.sort(
            key=lambda item: (item["sample_count"], item["purity"], -item["depth"]),
            reverse=True,
        )
        selected.extend(pool[:max_rules_per_class])
    return selected


def audit_representative_rules(
    model: DecisionTreeClassifier,
    rules: Iterable[dict[str, Any]],
    X_evaluation: Any,
    y_evaluation: Sequence[Any],
) -> list[dict[str, Any]]:
    """Measure each selected training leaf's support and purity out of sample."""

    leaves = model.apply(X_evaluation)
    targets = np.asarray(y_evaluation)
    if len(leaves) != len(targets):
        raise ValueError("Evaluation predictors and labels must have equal lengths.")
    audited: list[dict[str, Any]] = []
    for rule in rules:
        record = dict(rule)
        mask = leaves == int(record["leaf_id"])
        support = int(mask.sum())
        record["evaluation_sample_count"] = support
        record["evaluation_purity"] = (
            float(np.mean(targets[mask] == record["predicted_class_value"]))
            if support
            else None
        )
        audited.append(record)
    return audited


def write_representative_rules(
    rules: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write selected decision paths as a compact Markdown artifact."""

    lines = [
        "# Representative Baseline Decision Rules",
        "",
        (
            "These are selected high-support leaf paths from the fitted baseline. "
            "They are descriptive model rules, not causal statements."
        ),
        "",
    ]
    for index, rule in enumerate(rules, start=1):
        lines.extend(
            [
                f"## Rule {index}: predict `{rule['predicted_class']}`",
                "",
                *[f"- {condition}" for condition in rule["conditions"]],
                "",
                (
                    f"Leaf support: {rule['sample_count']:,} training rows; "
                    f"leaf purity: {rule['purity']:.3f}; path depth: {rule['depth']}."
                ),
                (
                    f"Held-out support: {rule.get('evaluation_sample_count', 0):,} rows; "
                    "held-out purity: "
                    + (
                        f"{rule['evaluation_purity']:.3f}."
                        if rule.get("evaluation_purity") is not None
                        else "not estimable (no matching rows)."
                    )
                ),
                "",
            ]
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
