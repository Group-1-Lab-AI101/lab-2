"""Kiet's cost-complexity pruning and split-criterion experiment.

The experiment keeps the group's official train/test split unchanged. Candidate
``ccp_alpha`` values are selected on an inner validation partition of the
official training data, so the held-out test labels do not influence tuning.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab2-matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

from src.artifact_utils import index_fingerprint, save_json
from src.baseline_tree import evaluate_classifier
from src.data import PROJECT_ROOT, load_and_split_data
from src.experiment_contract import (
    CLASS_DISPLAY_NAMES,
    FROZEN_BASELINE_PARAMETERS,
    POSITIVE_CLASS,
    POSITIVE_CLASS_NAME,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names


CRITERIA = ("gini", "entropy")
INNER_VALIDATION_SIZE = 0.20
MAX_ALPHA_CANDIDATES = 40

DEFAULT_FIGURES_OUTPUT = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TREES_OUTPUT = PROJECT_ROOT / "outputs" / "trees"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "kiet_pruning_and_criterion.md"


def tree_parameters(criterion: str, ccp_alpha: float) -> dict[str, Any]:
    """Build an explicit estimator configuration without mutating the baseline."""

    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}.")
    if not np.isfinite(ccp_alpha) or ccp_alpha < 0.0:
        raise ValueError("ccp_alpha must be a finite non-negative number.")

    parameters = dict(FROZEN_BASELINE_PARAMETERS)
    parameters.update(criterion=criterion, ccp_alpha=float(ccp_alpha))
    return parameters


def sample_ccp_alphas(
    ccp_alphas: Sequence[float],
    *,
    max_candidates: int = MAX_ALPHA_CANDIDATES,
) -> np.ndarray:
    """Select deterministic pruning-path candidates spanning small to large trees.

    The unpruned value ``0`` is always retained as a reference. The final path
    value, which normally collapses the model to a root-only stump, is excluded
    when intermediate positive alphas exist. Uniform path-index sampling gives
    coverage across the sequence of progressively smaller subtrees.
    """

    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2.")

    values = np.asarray(ccp_alphas, dtype=float)
    values = np.unique(values[np.isfinite(values) & (values >= 0.0)])
    if values.size == 0:
        raise ValueError("The pruning path did not contain a valid ccp_alpha.")

    positive = values[values > 0.0]
    if positive.size > 1:
        positive = positive[:-1]
    if positive.size <= max_candidates - 1:
        selected_positive = positive
    else:
        positions = np.linspace(0, positive.size - 1, max_candidates - 1)
        indices = np.unique(np.rint(positions).astype(int))
        selected_positive = positive[indices]

    return np.concatenate(([0.0], selected_positive)).astype(float)


def evaluate_pruning_candidates(
    X_fit: Any,
    y_fit: Sequence[Any],
    X_validation: Any,
    y_validation: Sequence[Any],
    *,
    criterion: str,
    max_candidates: int = MAX_ALPHA_CANDIDATES,
) -> pd.DataFrame:
    """Fit and score sampled subtrees for one split criterion."""

    unpruned = DecisionTreeClassifier(**tree_parameters(criterion, 0.0))
    path = unpruned.cost_complexity_pruning_path(X_fit, y_fit)
    alphas = sample_ccp_alphas(path.ccp_alphas, max_candidates=max_candidates)

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        model = DecisionTreeClassifier(**tree_parameters(criterion, float(alpha)))
        model.fit(X_fit, y_fit)
        fit_accuracy = accuracy_score(y_fit, model.predict(X_fit))
        validation_accuracy = accuracy_score(
            y_validation, model.predict(X_validation)
        )
        rows.append(
            {
                "criterion": criterion,
                "ccp_alpha": float(alpha),
                "fit_accuracy": float(fit_accuracy),
                "validation_accuracy": float(validation_accuracy),
                "validation_error_rate": float(1.0 - validation_accuracy),
                "train_validation_gap": float(fit_accuracy - validation_accuracy),
                "depth": int(model.get_depth()),
                "leaves": int(model.get_n_leaves()),
                "nodes": int(model.tree_.node_count),
            }
        )
    return pd.DataFrame(rows)


def select_best_pruned_candidate(candidates: pd.DataFrame) -> dict[str, Any]:
    """Choose the most accurate positive-alpha model, preferring simpler ties."""

    required = {
        "criterion",
        "ccp_alpha",
        "validation_accuracy",
        "leaves",
        "nodes",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")

    pruned = candidates.loc[candidates["ccp_alpha"] > 0.0].copy()
    if pruned.empty:
        raise ValueError("At least one positive ccp_alpha candidate is required.")
    ordered = pruned.sort_values(
        ["validation_accuracy", "leaves", "nodes", "ccp_alpha"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    record = ordered.iloc[0].to_dict()
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in record.items()
    }


def _model_record(
    *,
    model_name: str,
    model: DecisionTreeClassifier,
    criterion: str,
    ccp_alpha: float,
    validation_accuracy: float,
    metrics: dict[str, Any],
    baseline_leaves: int,
) -> dict[str, Any]:
    """Combine metrics, configuration, and tree size in one serializable row."""

    leaves = int(model.get_n_leaves())
    return {
        "model": model_name,
        "criterion": criterion,
        "ccp_alpha": float(ccp_alpha),
        "validation_accuracy": float(validation_accuracy),
        "accuracy": float(metrics["accuracy"]),
        "error_rate": float(metrics["error_rate"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "roc_auc": float(metrics["roc_auc"]),
        "train_accuracy": float(metrics["train_accuracy"]),
        "train_test_accuracy_gap": float(metrics["train_test_accuracy_gap"]),
        "depth": int(model.get_depth()),
        "leaves": leaves,
        "nodes": int(model.tree_.node_count),
        "leaf_reduction_vs_baseline": float(1.0 - leaves / baseline_leaves),
    }


def plot_pruning_tradeoff(candidates: pd.DataFrame, output_path: str | Path) -> None:
    """Plot validation performance against subtree size and pruning strength."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    colors = {"gini": "#2563eb", "entropy": "#dc2626"}
    for criterion in CRITERIA:
        group = candidates.loc[candidates["criterion"] == criterion].copy()
        group = group.sort_values("ccp_alpha")
        selected = select_best_pruned_candidate(group)
        label = criterion.title()
        color = colors[criterion]
        axes[0].plot(
            group["leaves"],
            group["validation_accuracy"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
            color=color,
            alpha=0.85,
        )
        axes[0].scatter(
            [selected["leaves"]],
            [selected["validation_accuracy"]],
            marker="*",
            s=170,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        axes[1].plot(
            group["ccp_alpha"],
            group["validation_accuracy"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
            color=color,
            alpha=0.85,
        )
        axes[1].scatter(
            [selected["ccp_alpha"]],
            [selected["validation_accuracy"]],
            marker="*",
            s=170,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of leaves (log scale)")
    axes[0].set_ylabel("Inner-validation accuracy")
    axes[0].set_title("Performance versus tree size")
    axes[0].grid(alpha=0.2)
    axes[0].legend(title="Criterion")

    axes[1].set_xscale("symlog", linthresh=1e-8)
    axes[1].set_xlabel("ccp_alpha (symlog scale)")
    axes[1].set_ylabel("Inner-validation accuracy")
    axes[1].set_title("Performance versus pruning strength")
    axes[1].grid(alpha=0.2)
    axes[1].legend(title="Criterion")

    figure.suptitle("Kiet - Cost-Complexity Pruning Trade-off", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_selected_tree(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    output_path: str | Path,
    *,
    criterion: str,
    ccp_alpha: float,
    max_depth: int = 3,
) -> None:
    """Save readable top levels of the selected, fully fitted pruned tree."""

    figure, axis = plt.subplots(figsize=(28, 14))
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=list(CLASS_DISPLAY_NAMES),
        filled=True,
        rounded=True,
        proportion=True,
        precision=3,
        max_depth=max_depth,
        fontsize=8,
        ax=axis,
    )
    axis.set_title(
        "Selected Pruned Decision Tree - "
        f"{criterion.title()}, ccp_alpha={ccp_alpha:.8g} "
        f"(displayed through depth {max_depth})",
        fontsize=17,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _format_test_rows(comparison: pd.DataFrame) -> str:
    return "\n".join(
        (
            f"| {row.model} | {row.criterion.title()} | {row.ccp_alpha:.8g} | "
            f"{row.accuracy:.6f} | {row.error_rate:.6f} | {row.precision:.6f} | "
            f"{row.recall:.6f} | {row.f1_score:.6f} | {row.roc_auc:.6f} | "
            f"{int(row.depth)} | {int(row.leaves):,} | {int(row.nodes):,} |"
        )
        for row in comparison.itertuples(index=False)
    )


def _effect_sentence(record: dict[str, Any], baseline: dict[str, Any]) -> str:
    delta = record["accuracy"] - baseline["accuracy"]
    direction = "tăng" if delta > 0 else "giảm" if delta < 0 else "không đổi"
    leaf_reduction = 100.0 * record["leaf_reduction_vs_baseline"]
    return (
        f"{record['model']} làm Accuracy {direction} {abs(delta):.6f} điểm so với "
        f"baseline, đồng thời giảm {leaf_reduction:.2f}% số lá "
        f"({baseline['leaves']:,} xuống {record['leaves']:,})."
    )


def render_pruning_report(
    *,
    report_path: Path,
    figures_dir: Path,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_by_criterion: dict[str, dict[str, Any]],
    selected_criterion: str,
    inner_fit_rows: int,
    inner_validation_rows: int,
) -> None:
    """Write Kiet's report-ready Vietnamese Markdown section from measured data."""

    relative_figures = Path(
        os.path.relpath(figures_dir.resolve(), start=report_path.parent.resolve())
    )
    baseline = comparison.loc[comparison["model"] == "Baseline (unpruned)"].iloc[0]
    baseline_record = baseline.to_dict()
    pruned_records = comparison.loc[comparison["model"].str.startswith("Pruned")]
    winner = comparison.loc[
        comparison["model"] == f"Pruned {selected_criterion.title()}"
    ].iloc[0]
    best_test = comparison.sort_values(
        ["accuracy", "f1_score"], ascending=[False, False], kind="mergesort"
    ).iloc[0]

    selection_rows = "\n".join(
        (
            f"| {criterion.title()} | {selection['ccp_alpha']:.8g} | "
            f"{selection['fit_accuracy']:.6f} | "
            f"{selection['validation_accuracy']:.6f} | "
            f"{selection['validation_error_rate']:.6f} | "
            f"{int(selection['depth'])} | {int(selection['leaves']):,} | "
            f"{int(selection['nodes']):,} |"
        )
        for criterion, selection in selected_by_criterion.items()
    )
    interpretation = "\n".join(
        f"- {_effect_sentence(row._asdict(), baseline_record)}"
        for row in pruned_records.itertuples(index=False)
    )
    candidate_counts = ", ".join(
        f"{criterion.title()}: {int((validation['criterion'] == criterion).sum())}"
        for criterion in CRITERIA
    )
    winner_delta = float(winner["accuracy"] - baseline["accuracy"])
    winner_direction = (
        "cao hơn" if winner_delta > 0 else "thấp hơn" if winner_delta < 0 else "bằng"
    )

    report = f"""# Improvement Methods - Phương pháp 2: Cost-Complexity Pruning

## Mục tiêu và nguyên lý

Phần này do **Thái Kiệt** phụ trách. Mục tiêu là giảm hiện tượng overfitting của cây baseline không giới hạn bằng Minimal Cost-Complexity Pruning và đồng thời so sánh hai tiêu chí tách `gini` và `entropy`. Thuật toán cân bằng sai số và độ phức tạp theo `R_alpha(T) = R(T) + alpha * |T|`; `ccp_alpha` càng lớn thì mức phạt cho số lá càng mạnh và cây càng nhỏ.

## Thiết kế thí nghiệm và chống rò rỉ dữ liệu

- Giữ nguyên dataset, target `no=0`/`yes=1`, stratified train/test 80%/20% và `random_state=42` của nhóm.
- Tập test chính thức gồm 9,043 dòng được giữ nguyên cho đánh giá cuối; không dùng để chọn `ccp_alpha` hay tiêu chí tách.
- Tập train chính thức được chia tiếp theo stratified split thành {inner_fit_rows:,} dòng inner-fit và {inner_validation_rows:,} dòng validation. Encoder được fit riêng trên inner-fit và validation chỉ được transform.
- Với từng criterion, lấy pruning path, giữ mốc `ccp_alpha=0` làm đối chứng và lấy mẫu tối đa {MAX_ALPHA_CANDIDATES} cấu hình trải trên đường cắt tỉa. Số cấu hình thực tế: {candidate_counts}.
- Cấu hình pruning của mỗi criterion được chọn bằng Accuracy validation cao nhất; nếu bằng nhau thì ưu tiên cây có ít lá/node hơn. Chỉ các ứng viên có `ccp_alpha > 0` mới được xem là mô hình pruned.
- Sau khi khóa `criterion` và `ccp_alpha`, mô hình được fit lại trên toàn bộ 36,168 dòng train rồi đánh giá trên test chính thức.

## Kết quả chọn `ccp_alpha` trên validation

| Criterion | Selected `ccp_alpha` | Inner-fit Accuracy | Validation Accuracy | Validation Error | Depth | Leaves | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{selection_rows}

![Quan hệ giữa pruning, kích thước cây và Accuracy]({relative_figures.as_posix()}/kiet_pruning_tradeoff.png)

Đồ thị cho thấy `ccp_alpha` làm giảm dần số lá và thường thu hẹp chênh lệch train-validation. Cắt tỉa quá ít vẫn giữ nhiều nhánh đặc thù của tập train; cắt tỉa quá mạnh làm mất các quy tắc hữu ích và gây underfitting. Dấu sao là cấu hình được chọn cho từng criterion.

## Kết quả trên tập test giữ lại

| Model | Criterion | `ccp_alpha` | Accuracy | Error Rate | Precision (`yes`) | Recall (`yes`) | F1 (`yes`) | ROC-AUC | Depth | Leaves | Nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_format_test_rows(comparison)}

Hai dòng unpruned tách riêng ảnh hưởng của criterion; hai dòng pruned cho thấy ảnh hưởng kết hợp của criterion và `ccp_alpha`. Accuracy và Error Rate được báo cáo theo yêu cầu đề bài, còn Precision/Recall/F1/ROC-AUC giúp tránh kết luận sai khi lớp `yes` là lớp thiểu số.

Xét riêng số liệu mô tả trên test, **{best_test['model']}** đạt Accuracy cao nhất ({float(best_test['accuracy']):.6f}) và Error Rate thấp nhất ({float(best_test['error_rate']):.6f}). Tuy nhiên, cấu hình chính vẫn được khóa bằng validation: Gini có Validation Accuracy {float(selected_by_criterion['gini']['validation_accuracy']):.6f} so với {float(selected_by_criterion['entropy']['validation_accuracy']):.6f} của Entropy. Không đổi lựa chọn sau khi xem test giúp tránh tối ưu gián tiếp trên tập test.

## Cây đã cắt tỉa được chọn

Tiêu chí cuối cùng được khóa theo validation là **{selected_criterion.title()}**, với `ccp_alpha={float(winner['ccp_alpha']):.8g}`. Hình dưới chỉ hiển thị các tầng đầu để đọc được nhãn; các thống kê depth/leaves/nodes trong bảng được tính trên toàn bộ cây đã fit.

![Các tầng đầu của cây đã cắt tỉa]({relative_figures.as_posix()}/kiet_pruned_tree_top_levels.png)

## Phân tích và giải thích kết quả

{interpretation}

- Cấu hình được chọn theo validation đạt Accuracy test {float(winner['accuracy']):.6f}, {winner_direction} baseline {abs(winner_delta):.6f} điểm; Error Rate là {float(winner['error_rate']):.6f}.
- Gini đo mức không thuần bằng `1 - sum(p_k^2)`, còn Entropy dùng `-sum(p_k * log2(p_k))`. Hai tiêu chí có thể chọn các split khác nhau, nên pruning path, `ccp_alpha` tối ưu và kích thước cây cũng khác nhau.
- Lợi ích chính của pruning không chỉ nằm ở Accuracy: cây nhỏ hơn giảm variance, chênh lệch train-test và chi phí diễn giải. Nếu Accuracy không tăng, kết quả vẫn cho biết mức đơn giản hóa đạt được và chỉ ra rằng cắt tỉa không tự động giải quyết mất cân bằng lớp.
- Kết luận chỉ áp dụng cho split và pipeline cố định của nhóm. Không chọn lại cấu hình bằng kết quả test để tránh test leakage.

## Tệp kết quả phục vụ ghép báo cáo

- `outputs/results/kiet_pruning_validation.csv`: toàn bộ điểm trên pruning path đã lấy mẫu.
- `outputs/results/kiet_pruning_test_comparison.csv`: bảng so sánh baseline, Gini/Entropy và pruning.
- `outputs/results/kiet_pruning_metrics.json`: cấu hình chọn, metric và audit split.
- `outputs/trees/kiet_pruned_tree_model.joblib`: mô hình pruned được chọn theo validation.
"""
    report_path.write_text(report, encoding="utf-8")


def run_pruning_workflow(
    *,
    figures_dir: str | Path = DEFAULT_FIGURES_OUTPUT,
    results_dir: str | Path = DEFAULT_RESULTS_OUTPUT,
    trees_dir: str | Path = DEFAULT_TREES_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    max_alpha_candidates: int = MAX_ALPHA_CANDIDATES,
) -> dict[str, Any]:
    """Run Kiet's complete pruning experiment and write reproducible artifacts."""

    resolved_figures = Path(figures_dir).resolve()
    resolved_results = Path(results_dir).resolve()
    resolved_trees = Path(trees_dir).resolve()
    resolved_report = Path(report_path).resolve()
    for directory in (
        resolved_figures,
        resolved_results,
        resolved_trees,
        resolved_report.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    split = load_and_split_data(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_inner_fit, X_validation, y_inner_fit, y_validation = train_test_split(
        split.X_train,
        split.y_train,
        test_size=INNER_VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=split.y_train,
    )

    inner_preprocessing = build_preprocessing_pipeline()
    X_inner_fit_processed = inner_preprocessing.fit_transform(X_inner_fit)
    X_validation_processed = inner_preprocessing.transform(X_validation)

    candidate_tables = []
    selected_by_criterion: dict[str, dict[str, Any]] = {}
    for criterion in CRITERIA:
        candidates = evaluate_pruning_candidates(
            X_inner_fit_processed,
            y_inner_fit,
            X_validation_processed,
            y_validation,
            criterion=criterion,
            max_candidates=max_alpha_candidates,
        )
        candidate_tables.append(candidates)
        selected_by_criterion[criterion] = select_best_pruned_candidate(candidates)

    validation = pd.concat(candidate_tables, ignore_index=True)
    validation.to_csv(resolved_results / "kiet_pruning_validation.csv", index=False)
    plot_pruning_tradeoff(validation, resolved_figures / "kiet_pruning_tradeoff.png")

    selected_criterion = min(
        CRITERIA,
        key=lambda name: (
            -selected_by_criterion[name]["validation_accuracy"],
            selected_by_criterion[name]["leaves"],
            selected_by_criterion[name]["nodes"],
            CRITERIA.index(name),
        ),
    )

    preprocessing = build_preprocessing_pipeline()
    X_train_processed = preprocessing.fit_transform(split.X_train)
    X_test_processed = preprocessing.transform(split.X_test)
    feature_names = get_encoded_feature_names(preprocessing)

    model_specs = [
        ("Baseline (unpruned)", "gini", 0.0),
        ("Unpruned Entropy", "entropy", 0.0),
        (
            "Pruned Gini",
            "gini",
            float(selected_by_criterion["gini"]["ccp_alpha"]),
        ),
        (
            "Pruned Entropy",
            "entropy",
            float(selected_by_criterion["entropy"]["ccp_alpha"]),
        ),
    ]
    fitted: dict[str, DecisionTreeClassifier] = {}
    evaluated: dict[str, dict[str, Any]] = {}
    reports: dict[str, pd.DataFrame] = {}
    validation_lookup = {
        (row.criterion, float(row.ccp_alpha)): float(row.validation_accuracy)
        for row in validation.itertuples(index=False)
    }

    for model_name, criterion, alpha in model_specs:
        model = DecisionTreeClassifier(**tree_parameters(criterion, alpha))
        model.fit(X_train_processed, split.y_train)
        metrics, classification, _ = evaluate_classifier(
            model,
            X_train_processed,
            split.y_train,
            X_test_processed,
            split.y_test,
            positive_label=POSITIVE_CLASS,
            positive_class_name=POSITIVE_CLASS_NAME,
            class_display_names=CLASS_DISPLAY_NAMES,
        )
        fitted[model_name] = model
        evaluated[model_name] = metrics
        reports[model_name] = classification

    baseline_leaves = fitted["Baseline (unpruned)"].get_n_leaves()
    records = []
    for model_name, criterion, alpha in model_specs:
        records.append(
            _model_record(
                model_name=model_name,
                model=fitted[model_name],
                criterion=criterion,
                ccp_alpha=alpha,
                validation_accuracy=validation_lookup[(criterion, alpha)],
                metrics=evaluated[model_name],
                baseline_leaves=baseline_leaves,
            )
        )
    comparison = pd.DataFrame(records)
    comparison.to_csv(
        resolved_results / "kiet_pruning_test_comparison.csv", index=False
    )

    selected_model_name = f"Pruned {selected_criterion.title()}"
    selected_model = fitted[selected_model_name]
    selected_alpha = float(selected_by_criterion[selected_criterion]["ccp_alpha"])
    joblib.dump(selected_model, resolved_trees / "kiet_pruned_tree_model.joblib")
    for criterion in CRITERIA:
        joblib.dump(
            fitted[f"Pruned {criterion.title()}"],
            resolved_trees / f"kiet_pruned_{criterion}_model.joblib",
        )
    plot_selected_tree(
        selected_model,
        feature_names,
        resolved_figures / "kiet_pruned_tree_top_levels.png",
        criterion=selected_criterion,
        ccp_alpha=selected_alpha,
    )
    export_graphviz(
        selected_model,
        out_file=str(resolved_trees / "kiet_pruned_tree_full.dot"),
        feature_names=list(feature_names),
        class_names=list(CLASS_DISPLAY_NAMES),
        filled=True,
        rounded=True,
        special_characters=False,
        precision=3,
    )

    for model_name, classification in reports.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        classification.to_csv(
            resolved_results / f"kiet_{safe_name}_classification_report.csv"
        )

    train_indices = split.X_train.index.to_numpy(dtype=np.int64, copy=True)
    test_indices = split.X_test.index.to_numpy(dtype=np.int64, copy=True)
    payload = {
        "scope": "Thai Kiet - cost-complexity pruning and Gini/Entropy comparison",
        "selection_policy": {
            "test_set_used_for_selection": False,
            "inner_validation_size": INNER_VALIDATION_SIZE,
            "inner_fit_rows": int(len(y_inner_fit)),
            "inner_validation_rows": int(len(y_validation)),
            "max_alpha_candidates_per_criterion": int(max_alpha_candidates),
            "rule": (
                "Highest inner-validation accuracy among ccp_alpha > 0; "
                "ties prefer fewer leaves, fewer nodes, then smaller ccp_alpha."
            ),
        },
        "split_audit": {
            "official_train_rows": int(len(split.y_train)),
            "official_test_rows": int(len(split.y_test)),
            "train_indices_sha256": index_fingerprint(train_indices),
            "test_indices_sha256": index_fingerprint(test_indices),
            "official_train_test_disjoint": bool(
                np.intersect1d(train_indices, test_indices).size == 0
            ),
            "inner_fit_validation_disjoint": bool(
                set(X_inner_fit.index).isdisjoint(X_validation.index)
            ),
            "test_partition_usage": "final evaluation only",
        },
        "selected_by_criterion": selected_by_criterion,
        "selected_criterion": selected_criterion,
        "selected_model": selected_model_name,
        "comparison": records,
    }
    save_json(payload, resolved_results / "kiet_pruning_metrics.json")

    render_pruning_report(
        report_path=resolved_report,
        figures_dir=resolved_figures,
        validation=validation,
        comparison=comparison,
        selected_by_criterion=selected_by_criterion,
        selected_criterion=selected_criterion,
        inner_fit_rows=len(y_inner_fit),
        inner_validation_rows=len(y_validation),
    )

    return {
        "selected_criterion": selected_criterion,
        "selected_ccp_alpha": selected_alpha,
        "selected_model": selected_model_name,
        "selected_metrics": evaluated[selected_model_name],
        "validation_candidates": int(len(validation)),
        "report": str(resolved_report),
        "comparison_csv": str(
            resolved_results / "kiet_pruning_test_comparison.csv"
        ),
        "tradeoff_figure": str(resolved_figures / "kiet_pruning_tradeoff.png"),
        "tree_figure": str(resolved_figures / "kiet_pruned_tree_top_levels.png"),
    }
