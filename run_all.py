"""Run the unified Khang, Hoang, Hau, and Kiet project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.baseline_workflow import (
    DEFAULT_REPORT as DEFAULT_BASELINE_REPORT,
    DEFAULT_RESULTS_OUTPUT,
    DEFAULT_SHARED_OUTPUT,
    DEFAULT_TREES_OUTPUT,
    run_baseline_workflow,
)
from src.data import (
    DEFAULT_DATA_PATH,
    load_bank_data,
    make_stratified_split,
    split_features_target,
)
from src.eda import build_eda_report, build_split_report
from src.experiment_contract import DATASET_PATH, RANDOM_STATE, TEST_SIZE
from src.hau_hyperparameter_tuning import (
    DEFAULT_REPORT as DEFAULT_HAU_REPORT,
    run_hau_hyperparameter_workflow,
)
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names
from src.pruning_experiment import (
    DEFAULT_REPORT as DEFAULT_KIET_REPORT,
    run_pruning_workflow,
)
from src.utils import to_pretty_json
from src.visualization import DEFAULT_FIGURES_DIR, generate_eda_figures


def build_report(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Recompute every EDA, split, and encoding value from the source CSV.

    The encoder is fitted exclusively on the training partition before both
    matrices are transformed. The function does not train a classifier and does
    not write files, making it safe to call from tests and notebooks.

    Args:
        data_path: Location of the semicolon-delimited Bank Marketing CSV.
        test_size: Fraction of observations reserved for the held-out test set.
        random_state: Seed controlling the stratified shuffled split.

    Returns:
        A JSON-compatible dictionary with ``eda``, ``split``, and ``encoding``
        sections, including transformed shapes and ordered feature names.

    Raises:
        FileNotFoundError: If ``data_path`` cannot be found.
        ValueError: If dataset validation, splitting, or preprocessing fails.
    """

    frame = load_bank_data(data_path)
    X, y = split_features_target(frame)
    split = make_stratified_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Fit only on training data to avoid leaking test-set category information.
    preprocessing_pipeline = build_preprocessing_pipeline()
    encoded_train = preprocessing_pipeline.fit_transform(split.X_train)
    encoded_test = preprocessing_pipeline.transform(split.X_test)
    feature_names = get_encoded_feature_names(preprocessing_pipeline)

    return {
        "eda": build_eda_report(frame),
        "split": build_split_report(split),
        "encoding": {
            "method": "OneHotEncoder(handle_unknown='ignore')",
            "fit_scope": "training set only",
            "target_mapping": {"no": 0, "yes": 1},
            "input_features": int(split.X_train.shape[1]),
            "encoded_features": len(feature_names),
            "encoded_train_shape": [
                int(encoded_train.shape[0]),
                int(encoded_train.shape[1]),
            ],
            "encoded_test_shape": [
                int(encoded_test.shape[0]),
                int(encoded_test.shape[1]),
            ],
            "feature_names": feature_names,
        },
    }


def build_figures(
    data_path: str | Path = DEFAULT_DATA_PATH,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Recreate all EDA figures from the same data and split configuration.

    Args:
        data_path: Location of the semicolon-delimited Bank Marketing CSV.
        test_size: Fraction of observations reserved for the held-out test set.
        random_state: Seed controlling the stratified shuffled split.
        output_dir: Directory that receives the four generated PNG files.

    Returns:
        Absolute paths to all generated figures in their report display order.

    Raises:
        FileNotFoundError: If ``data_path`` cannot be found.
        ValueError: If dataset validation or stratified splitting fails.
        OSError: If the figure directory or a PNG file cannot be written.
    """

    frame = load_bank_data(data_path)
    X, y = split_features_target(frame)
    split = make_stratified_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return generate_eda_figures(frame, split, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse options for the single end-to-end project workflow.

    Returns:
        An argparse namespace containing figure options and the unified output
        and report destinations.

    Raises:
        SystemExit: If command-line arguments are invalid or help was requested.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Khang's EDA/preprocessing, Hoang's frozen baseline, "
            "Hau's hyperparameter tuning, and Kiet's pruning experiment from "
            "one entry point."
        )
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Directory for EDA PNGs (default: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Print the JSON report without generating EDA figures",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT,
        help=f"Directory for metrics and audits (default: {DEFAULT_RESULTS_OUTPUT})",
    )
    parser.add_argument(
        "--trees-dir",
        type=Path,
        default=DEFAULT_TREES_OUTPUT,
        help=f"Directory for models and tree exports (default: {DEFAULT_TREES_OUTPUT})",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=DEFAULT_BASELINE_REPORT,
        help=f"Baseline Markdown report (default: {DEFAULT_BASELINE_REPORT})",
    )
    parser.add_argument(
        "--shared-output-dir",
        type=Path,
        default=DEFAULT_SHARED_OUTPUT,
        help=f"Directory for shared fitted outputs (default: {DEFAULT_SHARED_OUTPUT})",
    )
    parser.add_argument(
        "--hau-report",
        type=Path,
        default=DEFAULT_HAU_REPORT,
        help=f"Hau tuning Markdown report (default: {DEFAULT_HAU_REPORT})",
    )
    parser.add_argument(
        "--kiet-report",
        type=Path,
        default=DEFAULT_KIET_REPORT,
        help=f"Kiet pruning Markdown report (default: {DEFAULT_KIET_REPORT})",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the single project workflow and print one combined JSON summary.

    The workflow regenerates Khang's EDA and figures, trains and documents
    Hoang's frozen baseline, tunes Hau's separately owned tree using
    training-only cross-validation, then runs Kiet's pruning experiment with
    the same split and preprocessing contract. Exceptions propagate so failures
    return a non-zero exit status.

    Returns:
        None.
    """

    args = parse_args()
    khang_report = build_report(
        DATASET_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    if not args.skip_figures:
        figure_paths = build_figures(
            DATASET_PATH,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            output_dir=args.figures_dir,
        )
        khang_report["figures"] = [str(path) for path in figure_paths]

    baseline_report = run_baseline_workflow(
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        trees_dir=args.trees_dir,
        report_path=args.baseline_report,
        shared_output_dir=args.shared_output_dir,
    )
    hau_report = run_hau_hyperparameter_workflow(
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        trees_dir=args.trees_dir,
        report_path=args.hau_report,
        baseline_metrics=baseline_report["metrics"],
        baseline_complexity=baseline_report["tree_statistics"],
    )
    kiet_report = run_pruning_workflow(
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        trees_dir=args.trees_dir,
        report_path=args.kiet_report,
    )

    print(
        to_pretty_json(
            {
                "khang": khang_report,
                "baseline": baseline_report,
                "hau": hau_report,
                "kiet_pruning": kiet_report,
            }
        )
    )


if __name__ == "__main__":
    main()
