"""Timeframe filename encoding helpers (cross-platform safe)."""

from __future__ import annotations


def timeframe_to_file_token(timeframe: str) -> str:
    """Encode timeframe into a case-safe filename token."""

    tf = str(timeframe).strip()
    if tf == "1M":
        return "1mo"
    return tf


def file_token_to_timeframe(token: str) -> str:
    """Decode filename token back to canonical timeframe."""

    value = str(token).strip()
    if value == "1mo":
        return "1M"
    return value
