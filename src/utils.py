"""Small reporting helpers shared by command-line scripts."""

from __future__ import annotations

import json
from typing import Any


def to_pretty_json(value: dict[str, Any]) -> str:
    """Serialize a report as readable, deterministic, Unicode-preserving JSON.

    Args:
        value: JSON-compatible report dictionary. Nested values must be composed
            of mappings, sequences, strings, numbers, booleans, or ``None``.

    Returns:
        A two-space-indented JSON string that retains non-ASCII text and keeps
        dictionary insertion order for stable terminal output and diffs.

    Raises:
        TypeError: If ``value`` contains an object that the standard JSON encoder
            cannot serialize.
    """

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)
