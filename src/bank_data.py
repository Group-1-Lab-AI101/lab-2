"""Shared Bank Marketing split and preprocessing contract.

Khang has not implemented his owned data section yet.  This module therefore
provides the minimum shared infrastructure needed to run Hoang's baseline and
later experiments on exactly the same partition.  Khang may extend or replace
the implementation, but the contract and frozen split must remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.experiment_contract import (
    CLASS_LABELS,
    DATASET_PATH,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


# Backward-compatible names for callers written before the handoff contract.
EXPECTED_TARGET = TARGET_COLUMN
EXPECTED_LABELS = CLASS_LABELS


@dataclass(frozen=True)
class PreparedData:
    """The one reusable data bundle for every team member's experiment."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_processed: np.ndarray
    X_test_processed: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    train_row_indices: np.ndarray
    test_row_indices: np.ndarray
    feature_names: np.ndarray
    class_labels: np.ndarray
    positive_label: str
    preprocessor: ColumnTransformer
    raw_feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]


def load_bank_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load and validate UCI ``bank-full.csv``."""

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. See README.md for the official UCI download."
        )

    frame = pd.read_csv(path, sep=";")
    if EXPECTED_TARGET not in frame.columns:
        raise ValueError(
            f"Expected target column {EXPECTED_TARGET!r}; found {frame.columns.tolist()}"
        )
    if frame.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    observed_labels = set(frame[EXPECTED_TARGET].dropna().unique())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Expected binary labels {sorted(EXPECTED_LABELS)}; found {sorted(observed_labels)}"
        )
    return frame


def prepare_bank_data(frame: pd.DataFrame) -> PreparedData:
    """Apply the frozen split, then fit preprocessing on training data only."""

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} is missing.")

    X = frame.drop(columns=[TARGET_COLUMN]).copy()
    y = frame[TARGET_COLUMN].copy()
    if TARGET_COLUMN in X.columns:
        raise AssertionError("Target leakage: target is still present in predictors.")
    if y.isna().any():
        raise ValueError("Target contains missing values.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_row_indices = X_train_raw.index.to_numpy(dtype=np.int64, copy=True)
    test_row_indices = X_test_raw.index.to_numpy(dtype=np.int64, copy=True)
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    categorical = tuple(
        X_train_raw.select_dtypes(include=["object", "string", "category"]).columns
    )
    numeric = tuple(column for column in X_train_raw.columns if column not in categorical)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
            ("numeric", "passthrough", list(numeric)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Deliberately fit on X_train_raw only; the test partition is transform-only.
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
    X_test_processed = np.asarray(preprocessor.transform(X_test_raw), dtype=float)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)

    if (
        X_train_processed.shape[1] != len(feature_names)
        or X_test_processed.shape[1] != len(feature_names)
    ):
        raise AssertionError("Transformed columns and feature names are misaligned.")
    if TARGET_COLUMN in set(feature_names):
        raise AssertionError("Target leakage: transformed features contain the target.")
    if np.intersect1d(train_row_indices, test_row_indices).size:
        raise AssertionError("Train/test leakage: row indices overlap.")

    class_labels = np.sort(y.unique())
    return PreparedData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        train_row_indices=train_row_indices,
        test_row_indices=test_row_indices,
        feature_names=feature_names,
        class_labels=class_labels,
        positive_label=POSITIVE_CLASS,
        preprocessor=preprocessor,
        raw_feature_names=tuple(X.columns),
        categorical_features=categorical,
        numeric_features=numeric,
    )


def load_shared_experiment_data(dataset_path: str | Path = DATASET_PATH) -> PreparedData:
    """Load the dataset and return the frozen shared experiment bundle."""

    return prepare_bank_data(load_bank_dataset(dataset_path))


def preprocessing_audit(prepared: PreparedData) -> dict[str, Any]:
    """Return explicit, machine-readable leakage and alignment checks."""

    return {
        "split_before_preprocessor_fit": True,
        "preprocessor_fit_partition": "training only",
        "test_partition_usage": "transform and evaluation only",
        "target_column": TARGET_COLUMN,
        "target_excluded_from_raw_features": TARGET_COLUMN not in prepared.raw_feature_names,
        "target_excluded_from_transformed_features": TARGET_COLUMN
        not in set(prepared.feature_names),
        "target_excluded_from_raw_partitions": TARGET_COLUMN
        not in prepared.X_train_raw.columns
        and TARGET_COLUMN not in prepared.X_test_raw.columns,
        "train_test_indices_disjoint": bool(
            np.intersect1d(prepared.train_row_indices, prepared.test_row_indices).size == 0
        ),
        "split_row_count": int(
            len(prepared.train_row_indices) + len(prepared.test_row_indices)
        ),
        "feature_name_count": int(len(prepared.feature_names)),
        "transformed_train_column_count": int(prepared.X_train_processed.shape[1]),
        "transformed_test_column_count": int(prepared.X_test_processed.shape[1]),
        "feature_names_aligned": bool(
            prepared.X_train_processed.shape[1]
            == prepared.X_test_processed.shape[1]
            == len(prepared.feature_names)
        ),
    }
