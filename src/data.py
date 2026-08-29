"""Load, validate, and split the Bank Marketing dataset.

The functions in this module are the single source of truth for every group
experiment.  In particular, the split is stratified by the binary target and
is reproducible through ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bank-full.csv"
TARGET_COLUMN = "y"
TARGET_MAPPING = {"no": 0, "yes": 1}

NUMERICAL_FEATURES = (
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
)

CATEGORICAL_FEATURES = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
)

# Keep the original column order explicit so malformed input fails early.
EXPECTED_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    TARGET_COLUMN,
)

# Domains documented by the UCI Bank Marketing data dictionary. ``unknown`` is
# an observed, meaningful level and must not be treated as a missing value.
CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    "job": frozenset(
        {
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown",
        }
    ),
    "marital": frozenset({"divorced", "married", "single"}),
    "education": frozenset({"primary", "secondary", "tertiary", "unknown"}),
    "default": frozenset({"no", "yes"}),
    "housing": frozenset({"no", "yes"}),
    "loan": frozenset({"no", "yes"}),
    "contact": frozenset({"cellular", "telephone", "unknown"}),
    "month": frozenset(
        {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
    ),
    "poutcome": frozenset({"failure", "other", "success", "unknown"}),
}

# Inclusive bounds encode semantic constraints, not sample-specific extrema.
# ``balance`` is intentionally unbounded because overdrafts are legitimate.
NUMERICAL_BOUNDS: dict[str, tuple[int | None, int | None]] = {
    "age": (0, 120),
    "balance": (None, None),
    "day": (1, 31),
    "duration": (0, None),
    "campaign": (1, None),
    "pdays": (-1, None),
    "previous": (0, None),
}


@dataclass(frozen=True)
class DatasetSplit:
    """Hold one reproducible train/test partition of the dataset.

    Attributes:
        X_train: Predictor rows assigned to the training partition. Original
            DataFrame indices are retained so partition overlap can be audited.
        X_test: Predictor rows assigned to the held-out test partition.
        y_train: Binary training labels, encoded as ``no=0`` and ``yes=1``.
        y_test: Binary held-out labels using the same target mapping.
        test_size: Requested proportion of observations assigned to the test
            partition.
        random_state: Seed passed to scikit-learn when shuffling and splitting
            the observations.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_size: float
    random_state: int


def validate_bank_data_frame(frame: pd.DataFrame) -> None:
    """Reject malformed values even when a CSV happens to retain the schema.

    The checks cover row identity, missing/duplicate observations, feature
    dtypes, finite numerical values, documented categorical domains, semantic
    numerical bounds, and the binary target. The function does not mutate the
    supplied frame and is public so tests can exercise individual failure modes.
    """

    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset schema. "
            f"Expected {list(EXPECTED_COLUMNS)}, got {frame.columns.tolist()}."
        )
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if not frame.index.is_unique:
        raise ValueError("Dataset row indices must be unique.")
    if frame.duplicated().any():
        raise ValueError(
            f"Dataset contains {int(frame.duplicated().sum())} duplicate row(s)."
        )
    missing = frame.isna().sum()
    if int(missing.sum()) > 0:
        details = {name: int(count) for name, count in missing.items() if count}
        raise ValueError(f"Dataset contains missing values: {details}.")

    for feature in NUMERICAL_FEATURES:
        series = frame[feature]
        if not pd.api.types.is_integer_dtype(series.dtype):
            raise ValueError(
                f"Numerical feature '{feature}' must have an integer dtype; "
                f"found {series.dtype}."
            )
        values = series.to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Numerical feature '{feature}' contains non-finite values.")
        lower, upper = NUMERICAL_BOUNDS[feature]
        if lower is not None and bool((series < lower).any()):
            raise ValueError(f"Numerical feature '{feature}' must be >= {lower}.")
        if upper is not None and bool((series > upper).any()):
            raise ValueError(f"Numerical feature '{feature}' must be <= {upper}.")

    for feature in CATEGORICAL_FEATURES:
        series = frame[feature]
        if not pd.api.types.is_string_dtype(series.dtype):
            raise ValueError(
                f"Categorical feature '{feature}' must have a string dtype; "
                f"found {series.dtype}."
            )
        unexpected = sorted(set(series.astype(str)).difference(CATEGORICAL_DOMAINS[feature]))
        if unexpected:
            raise ValueError(
                f"Categorical feature '{feature}' contains values outside its "
                f"documented domain: {unexpected}."
            )

    labels = set(frame[TARGET_COLUMN].astype(str).unique())
    expected_labels = set(TARGET_MAPPING)
    if labels != expected_labels:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain exactly "
            f"{sorted(expected_labels)}; found {sorted(labels)}."
        )


def load_bank_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the semicolon-delimited Bank Marketing dataset and validate it.

    Validation is deliberately performed before any EDA or modeling so that a
    wrong delimiter, missing column, reordered schema, empty file, or malformed
    target cannot silently propagate into group experiments. Feature values are
    returned unchanged; in particular, the literal ``"unknown"`` remains a
    legitimate categorical level rather than being converted to a missing value.

    Args:
        path: Location of ``bank-full.csv``. Relative paths are resolved from the
            caller's current working directory. The project dataset is used by
            default.

    Returns:
        A DataFrame containing 16 predictors and the original string target
        column ``y`` in the source column order.

    Raises:
        FileNotFoundError: If ``path`` does not identify a regular file.
        ValueError: If schema, row identity, completeness, dtypes, numerical
            constraints, categorical domains, duplicates, or target labels fail
            the documented dataset contract.
    """

    data_path = Path(path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path, sep=";")
    validate_bank_data_frame(frame)

    return frame


def split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors from the target and encode ``no/yes`` as ``0/1``.

    The input DataFrame is not modified. A copy of the predictor columns is
    returned, while the target is mapped to the compact ``int8`` dtype and keeps
    its original index for alignment with the predictors.

    Args:
        frame: Dataset containing a target column named :data:`TARGET_COLUMN`.

    Returns:
        A two-item tuple ``(X, y)`` where ``X`` contains the unencoded predictor
        columns and ``y`` contains binary labels with the name ``y``.

    Raises:
        ValueError: If the target column is absent or contains a value outside
            the mapping ``{"no": 0, "yes": 1}``.
    """

    if TARGET_COLUMN not in frame:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    X = frame.drop(columns=TARGET_COLUMN).copy()
    y = frame[TARGET_COLUMN].map(TARGET_MAPPING)
    if y.isna().any():
        invalid = sorted(frame.loc[y.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unrecognized target labels: {invalid}")

    y = y.astype("int8").rename(TARGET_COLUMN)
    return X, y


def make_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a deterministic train/test partition stratified by target class.

    Stratification keeps the minority-class proportion nearly identical in the
    training and held-out sets. scikit-learn shuffles the observations using the
    supplied seed; original indices are intentionally preserved in both outputs.

    Args:
        X: Predictor matrix whose rows correspond one-to-one with ``y``.
        y: Target labels used both for stratification and as returned labels.
        test_size: Fraction of all observations reserved for testing. Must be
            strictly between zero and one.
        random_state: Seed controlling the shuffled split, allowing every team
            experiment to reconstruct the same partitions.

    Returns:
        A :class:`DatasetSplit` containing train/test predictors, labels, and the
        split configuration.

    Raises:
        ValueError: If ``test_size`` is invalid, ``X`` and ``y`` have different
            row counts, or fewer than two target classes are present.
        ValueError: If scikit-learn cannot stratify because a class has too few
            observations for the requested partition sizes.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if y.nunique() < 2:
        raise ValueError("Stratification requires at least two target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_size=test_size,
        random_state=random_state,
    )


def load_and_split_data(
    path: str | Path = DEFAULT_DATA_PATH,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplit:
    """Load, encode the target, and stratify the dataset in one shared call.

    This is the recommended entry point for model experiments because it applies
    the same schema validation, target mapping, split ratio, and random seed for
    every team member.

    Args:
        path: Location of the semicolon-delimited Bank Marketing CSV.
        test_size: Fraction of rows assigned to the held-out test partition.
        random_state: Seed controlling the shuffled stratified split.

    Returns:
        A validated :class:`DatasetSplit` ready for a preprocessing or model
        pipeline.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If dataset validation, target encoding, or stratification
            fails. See :func:`load_bank_data` and :func:`make_stratified_split`.
    """

    frame = load_bank_data(path)
    X, y = split_features_target(frame)
    return make_stratified_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
