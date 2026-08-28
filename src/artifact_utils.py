"""Small artifact and reproducibility utilities shared across experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a deterministic, human-readable JSON artifact."""

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

