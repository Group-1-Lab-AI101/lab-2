"""Reproducible exploratory summaries for Khang's assignment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    DatasetSplit,
)


def _target_distribution(target: pd.Series) -> dict[str, dict[str, float | int]]:
    """Calculate counts and percentages for every observed target class.

    Args:
        target: One-dimensional target values. Labels may be strings (the source
            ``no/yes`` values) or integers (the encoded ``0/1`` values).

    Returns:
        A dictionary keyed by the string representation of each sorted class.
        Every class contains an integer ``count`` and a ``percentage`` rounded
        to four decimal places.
    """

    counts = target.value_counts().sort_index()
    percentages = target.value_counts(normalize=True).sort_index().mul(100.0)
    return {
        str(label): {
            "count": int(counts[label]),
            "percentage": round(float(percentages[label]), 4),
        }
        for label in counts.index
    }


def build_eda_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Build the complete reproducible EDA summary used in Khang's report.

    The report covers shape, missingness, duplicates, target balance,
    categorical cardinalities, explicit ``"unknown"`` categories, and selected
    descriptive statistics for every numerical feature. Values are converted to
    standard Python scalars so the result can be serialized directly as JSON.

    Args:
        frame: Validated Bank Marketing DataFrame containing all expected
            numerical features, categorical features, and the target ``y``.

    Returns:
        A nested dictionary with ``dataset``, ``target``,
        ``categorical_cardinality``, ``explicit_unknown_counts``, and
        ``numeric_summary`` sections.

    Raises:
        KeyError: If any expected feature or the target column is absent.
        TypeError: If a numerical feature cannot be summarized numerically.
    """

    missing_by_column = frame.isna().sum()
    categorical_frame = frame.loc[:, list(CATEGORICAL_FEATURES)]
    numeric_summary = frame.loc[:, list(NUMERICAL_FEATURES)].describe().T

    return {
        "dataset": {
            "rows": int(frame.shape[0]),
            "columns_including_target": int(frame.shape[1]),
            "input_features": int(frame.shape[1] - 1),
            "numerical_features": len(NUMERICAL_FEATURES),
            "categorical_features": len(CATEGORICAL_FEATURES),
            "missing_values": int(missing_by_column.sum()),
            "missing_by_column": {
                column: int(value) for column, value in missing_by_column.items()
            },
            "duplicate_rows": int(frame.duplicated().sum()),
        },
        "target": _target_distribution(frame[TARGET_COLUMN]),
        "categorical_cardinality": {
            column: int(categorical_frame[column].nunique(dropna=False))
            for column in CATEGORICAL_FEATURES
        },
        "explicit_unknown_counts": {
            column: int(categorical_frame[column].eq("unknown").sum())
            for column in CATEGORICAL_FEATURES
        },
        "numeric_summary": {
            column: {
                "min": float(numeric_summary.loc[column, "min"]),
                "mean": round(float(numeric_summary.loc[column, "mean"]), 3),
                "median": float(numeric_summary.loc[column, "50%"]),
                "max": float(numeric_summary.loc[column, "max"]),
            }
            for column in NUMERICAL_FEATURES
        },
    }


def build_split_report(split: DatasetSplit) -> dict[str, Any]:
    """Summarize partition sizes and class proportions for a dataset split.

    Args:
        split: Train/test partition produced by
            :func:`src.data.make_stratified_split`.

    Returns:
        A JSON-compatible dictionary recording the split method, configuration,
        row counts, and per-class counts and percentages for train and test.
    """

    return {
        "method": "train_test_split(stratify=y)",
        "test_size": split.test_size,
        "random_state": split.random_state,
        "train": {
            "rows": int(len(split.X_train)),
            "target": _target_distribution(split.y_train),
        },
        "test": {
            "rows": int(len(split.X_test)),
            "target": _target_distribution(split.y_test),
        },
    }
