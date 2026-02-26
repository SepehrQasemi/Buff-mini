"""Stable hashing helpers used for reproducibility."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(payload: Any, length: int = 12) -> str:
    """Return a deterministic hash for JSON-serializable payloads."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
