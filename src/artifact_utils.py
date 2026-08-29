"""Small artifact and reproducibility utilities shared across experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_IDENTITY_KEYS = (
    "dataset_sha256",
    "train_indices_sha256",
    "test_indices_sha256",
    "test_size",
    "random_state",
    "target_mapping",
)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_fingerprint(indices: np.ndarray) -> str:
    """Create a stable identity for an ordered train or test row selection."""

    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_experiment_identity(
    dataset_path: str | Path,
    train_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
    *,
    test_size: float,
    random_state: int,
    target_mapping: Mapping[str, int],
) -> dict[str, Any]:
    """Build the canonical identity shared by comparable model artifacts."""

    return {
        "dataset_sha256": file_sha256(dataset_path),
        "train_indices_sha256": index_fingerprint(np.asarray(train_indices)),
        "test_indices_sha256": index_fingerprint(np.asarray(test_indices)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "target_mapping": {
            str(label): int(value) for label, value in target_mapping.items()
        },
    }


def validate_experiment_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any] | None,
    *,
    artifact_name: str,
) -> None:
    """Reject an artifact that was produced from another dataset or split."""

    if actual is None:
        raise RuntimeError(
            f"{artifact_name} does not contain an experiment_identity record. "
            "Regenerate all artifacts with run_all.py before comparing models."
        )
    missing = [key for key in EXPERIMENT_IDENTITY_KEYS if key not in actual]
    if missing:
        raise RuntimeError(
            f"{artifact_name} experiment identity is missing fields: {missing}."
        )

    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in EXPERIMENT_IDENTITY_KEYS
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise RuntimeError(
            f"{artifact_name} was generated from an incompatible experiment: "
            f"{mismatches}. Regenerate the workflow artifacts before comparison."
        )


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a deterministic, human-readable JSON artifact."""

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
