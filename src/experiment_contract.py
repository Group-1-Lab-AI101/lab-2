"""Frozen constants shared by every Lab 2 model experiment.

Khang owns the final dataset/preprocessing implementation.  Until that work is
merged, this module is the single source of truth that keeps Hoang's baseline
and all later experiments on the same data partition.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from src.data import DEFAULT_DATA_PATH, PROJECT_ROOT, TARGET_COLUMN, TARGET_MAPPING


DATASET_PATH = DEFAULT_DATA_PATH
DATASET_DISPLAY_PATH = DEFAULT_DATA_PATH.relative_to(PROJECT_ROOT)
POSITIVE_CLASS = TARGET_MAPPING["yes"]
POSITIVE_CLASS_NAME = "yes"
CLASS_DISPLAY_NAMES = tuple(
    label for label, _ in sorted(TARGET_MAPPING.items(), key=lambda item: item[1])
)

TEST_SIZE = 0.20
RANDOM_STATE = 42
SPLIT_STRATEGY = "stratified random train/test split"

# Every constructor argument is explicit so a library-default change cannot
# silently redefine the baseline.  This mapping must not be edited by later
# improvement experiments; they should construct their own estimators.
FROZEN_BASELINE_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None,
        "random_state": RANDOM_STATE,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "monotonic_cst": None,
    }
)
