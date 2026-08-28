"""Reproduce Khang's EDA, stratified split, and categorical encoding report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data import (
    DEFAULT_DATA_PATH,
    load_bank_data,
    make_stratified_split,
    split_features_target,
)
from src.eda import build_eda_report, build_split_report
from src.preprocessing import build_preprocessing_pipeline, get_encoded_feature_names
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
    """Parse command-line options for the reproducible Khang report.

    Returns:
        An argparse namespace containing the dataset path, split configuration,
        figure output directory, and optional figure-suppression flag.

    Raises:
        SystemExit: If command-line arguments are invalid or help was requested.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Print the reproducible report for EDA, stratified train/test split, "
            "and categorical encoding."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the semicolon-delimited CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved for testing (default: 0.20)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used by the stratified split (default: 42)",
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
    return parser.parse_args()


def main() -> None:
    """Execute the command-line report workflow and print formatted JSON.

    The function reads arguments from ``sys.argv``, validates and processes the
    dataset, generates four EDA figures unless disabled, and writes the report to
    standard output. Exceptions propagate so failures return a non-zero exit
    status.

    Returns:
        None.
    """

    args = parse_args()
    report = build_report(
        args.data,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    if not args.skip_figures:
        figure_paths = build_figures(
            args.data,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.figures_dir,
        )
        report["figures"] = [str(path) for path in figure_paths]
    print(to_pretty_json(report))


if __name__ == "__main__":
    main()
