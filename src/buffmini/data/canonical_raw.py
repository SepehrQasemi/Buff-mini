"""Shared helpers for Stage-26.9 canonical raw data workflows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.loader import standardize_ohlcv_frame, validate_ohlcv_frame


@dataclass(frozen=True)
class GapStats:
    gaps_detected: int
    max_gap_minutes: int


def symbol_safe(symbol: str) -> str:
    return str(symbol).replace("/", "-").replace(":", "-")


def raw_path(*, data_dir: Path = RAW_DATA_DIR, exchange: str, symbol: str, timeframe: str) -> Path:
    return Path(data_dir) / str(exchange).strip().lower() / symbol_safe(symbol) / f"{timeframe}.parquet"


def raw_meta_path(*, data_dir: Path = RAW_DATA_DIR, exchange: str, symbol: str, timeframe: str) -> Path:
    return raw_path(data_dir=data_dir, exchange=exchange, symbol=symbol, timeframe=timeframe).with_suffix(".meta.json")


def legacy_flat_raw_path(*, data_dir: Path = RAW_DATA_DIR, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol_safe(symbol)
    return Path(data_dir) / f"{safe_symbol}_{timeframe}.parquet"


def resolve_raw_path(*, data_dir: Path = RAW_DATA_DIR, exchange: str, symbol: str, timeframe: str) -> Path:
    nested = raw_path(data_dir=data_dir, exchange=exchange, symbol=symbol, timeframe=timeframe)
    if nested.exists():
        return nested
    flat = legacy_flat_raw_path(data_dir=data_dir, symbol=symbol, timeframe=timeframe)
    return flat


def resolve_raw_meta_path(*, data_dir: Path = RAW_DATA_DIR, exchange: str, symbol: str, timeframe: str) -> Path:
    nested = raw_meta_path(data_dir=data_dir, exchange=exchange, symbol=symbol, timeframe=timeframe)
    if nested.exists():
        return nested
    flat = legacy_flat_raw_path(data_dir=data_dir, symbol=symbol, timeframe=timeframe).with_suffix(".meta.json")
    return flat


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def detect_gaps_minutes(ts: pd.Series, *, expected_minutes: int = 1) -> GapStats:
    stamps = pd.to_datetime(ts, utc=True, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if len(stamps) <= 1:
        return GapStats(gaps_detected=0, max_gap_minutes=0)
    diffs = stamps.diff().dropna().dt.total_seconds() / 60.0
    gaps = diffs[diffs > float(expected_minutes)]
    if gaps.empty:
        return GapStats(gaps_detected=0, max_gap_minutes=0)
    max_gap = int(round(float(gaps.max())))
    return GapStats(gaps_detected=int(gaps.shape[0]), max_gap_minutes=max_gap)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = standardize_ohlcv_frame(frame)
    now = pd.Timestamp.now(tz="UTC")
    out = out.loc[pd.to_datetime(out["timestamp"], utc=True, errors="coerce") <= now].copy()
    out = standardize_ohlcv_frame(out)
    validate_ohlcv_frame(out)
    return out
