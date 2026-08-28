"""Reusable categorical encoding and preprocessing pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(
    *,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    numerical_features: Sequence[str] = NUMERICAL_FEATURES,
) -> ColumnTransformer:
    """Create the shared, unfitted transformer for all predictor columns.

    Categorical variables are one-hot encoded. Unknown levels encountered at
    inference time are ignored instead of causing a failure. Numerical columns
    pass through unchanged because decision trees do not require scaling.

    Args:
        categorical_features: Column names to one-hot encode. The nine Bank
            Marketing categorical predictors are used by default.
        numerical_features: Column names to pass through without scaling. The
            seven numerical predictors are used by default.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`. Its transformed
        output remains sparse whenever possible, and output feature names omit
        transformer prefixes for readable tree and feature-importance reports.

    Raises:
        ValueError: If a column name appears in both feature groups. Additional
            missing-column or invalid-dtype errors may be raised by scikit-learn
            when the transformer is fitted.
    """

    categorical_columns = list(categorical_features)
    numerical_columns = list(numerical_features)
    overlap = set(categorical_columns) & set(numerical_columns)
    if overlap:
        raise ValueError(f"Columns cannot be both categorical and numeric: {overlap}")

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
                categorical_columns,
            ),
            ("numerical", "passthrough", numerical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline(
    *,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    numerical_features: Sequence[str] = NUMERICAL_FEATURES,
) -> Pipeline:
    """Wrap the shared transformer in a preprocessing-only pipeline.

    This variant is useful for inspecting the encoded matrix and feature names
    without fitting a classifier. It must be fitted on training data before
    calling ``transform`` or :func:`get_encoded_feature_names`.

    Args:
        categorical_features: Column names sent to ``OneHotEncoder``.
        numerical_features: Column names passed through without modification.

    Returns:
        An unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline` containing a
        single step named ``preprocessor``.

    Raises:
        ValueError: If the categorical and numerical feature groups overlap.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                ),
            )
        ]
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    *,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
    numerical_features: Sequence[str] = NUMERICAL_FEATURES,
) -> Pipeline:
    """Attach an estimator to the shared preprocessing transformer.

    Keeping preprocessing and modeling in one pipeline prevents leakage during
    cross-validation: the one-hot encoder is refitted only on each training fold
    and is then applied to that fold's validation observations.

    Args:
        estimator: Unfitted scikit-learn-compatible classifier or estimator to
            place in the final ``model`` step.
        categorical_features: Column names sent to ``OneHotEncoder``.
        numerical_features: Column names passed through without scaling.

    Returns:
        An unfitted two-step pipeline named ``preprocessor`` and ``model`` that
        exposes the normal scikit-learn ``fit``/``predict`` interface.

    Raises:
        ValueError: If ``estimator`` is ``None`` or the feature groups overlap.
        TypeError: If the supplied object does not implement the estimator API;
            scikit-learn normally raises this when the pipeline is used.
    """

    if estimator is None:
        raise ValueError("estimator must be a scikit-learn estimator, not None.")

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    categorical_features=categorical_features,
                    numerical_features=numerical_features,
                ),
            ),
            ("model", estimator),
        ]
    )


def get_encoded_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Extract ordered transformed-feature names from a fitted pipeline.

    Args:
        fitted_pipeline: A preprocessing-only or model pipeline whose
            ``preprocessor`` step has already been fitted.

    Returns:
        Feature names in exactly the same order as columns in the transformed
        matrix: one-hot features first, followed by numerical passthrough fields.

    Raises:
        ValueError: If the pipeline has no step named ``preprocessor``.
        sklearn.exceptions.NotFittedError: If the preprocessor has not been
            fitted yet.
    """

    if "preprocessor" not in fitted_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'preprocessor' step.")
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()
