"""Frozen constants shared by every Lab 2 model experiment.

Khang owns the final dataset/preprocessing implementation.  Until that work is
merged, this module is the single source of truth that keeps Hoang's baseline
and all later experiments on the same data partition.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DATASET_PATH = Path("data/raw/bank-full.csv")
TARGET_COLUMN = "y"
POSITIVE_CLASS = "yes"
CLASS_LABELS = frozenset({"no", "yes"})

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

