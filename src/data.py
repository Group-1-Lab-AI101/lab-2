"""Load, validate, and split the Bank Marketing dataset.

The functions in this module are the single source of truth for every group
experiment.  In particular, the split is stratified by the binary target and
is reproducible through ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        ValueError: If the CSV is empty, its schema differs from
            :data:`EXPECTED_COLUMNS`, the target contains missing values, or the
            target labels are not exactly ``"no"`` and ``"yes"``.
    """

    data_path = Path(path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path, sep=";")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset schema. "
            f"Expected {list(EXPECTED_COLUMNS)}, got {frame.columns.tolist()}."
        )
    if frame.empty:
        raise ValueError("The dataset is empty.")

    labels = set(frame[TARGET_COLUMN].dropna().unique())
    expected_labels = set(TARGET_MAPPING)
    if labels != expected_labels or frame[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain only {sorted(expected_labels)}. "
            f"Found {sorted(labels)} and "
            f"{int(frame[TARGET_COLUMN].isna().sum())} missing value(s)."
        )

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
