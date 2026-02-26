"""Time helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def parse_utc_timestamp(value: str | None) -> pd.Timestamp | None:
    """Parse a timestamp into UTC pandas Timestamp."""

    if value is None:
        return None
    return pd.Timestamp(value, tz="UTC")


def utc_now_compact() -> str:
    """Return UTC time formatted for run IDs."""

    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
