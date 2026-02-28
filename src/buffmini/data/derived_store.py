"""Derived (non-OHLCV) parquet storage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import DERIVED_DATA_DIR


def symbol_safe(symbol: str) -> str:
    """Convert exchange symbol to filesystem-safe form."""

    return str(symbol).replace("/", "-").replace(":", "-")


def derived_parquet_path(
    kind: str,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> Path:
    """Return parquet path under data/derived/<kind>/ for one series."""

    return Path(data_dir) / str(kind) / f"{symbol_safe(symbol)}_{timeframe}.parquet"


def meta_json_path(
    kind: str,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> Path:
    """Return metadata JSON path under data/derived/_meta/."""

    return Path(data_dir) / "_meta" / f"{kind}_{symbol_safe(symbol)}_{timeframe}_meta.json"


def save_derived_parquet(
    frame: pd.DataFrame,
    kind: str,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> Path:
    """Save derived timeseries parquet."""

    path = derived_parquet_path(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_derived_parquet(
    kind: str,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> pd.DataFrame:
    """Load derived timeseries parquet."""

    path = derived_parquet_path(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Derived parquet not found: {path}")
    frame = pd.read_parquet(path)
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def write_meta_json(
    kind: str,
    symbol: str,
    timeframe: str,
    payload: dict[str, Any],
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> Path:
    """Write metadata JSON for a derived series."""

    path = meta_json_path(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def read_meta_json(
    kind: str,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Read metadata JSON for a derived series."""

    path = meta_json_path(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Derived metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def coverage(
    kind: str,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Return simple coverage info for one derived series."""

    path = derived_parquet_path(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if not path.exists():
        return {
            "kind": kind,
            "symbol": symbol,
            "timeframe": timeframe,
            "exists": False,
            "rows": 0,
            "start_ts": None,
            "end_ts": None,
            "path": str(path),
        }
    frame = load_derived_parquet(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if frame.empty:
        return {
            "kind": kind,
            "symbol": symbol,
            "timeframe": timeframe,
            "exists": True,
            "rows": 0,
            "start_ts": None,
            "end_ts": None,
            "path": str(path),
        }
    ts_col = "ts" if "ts" in frame.columns else "timestamp"
    ts = pd.to_datetime(frame[ts_col], utc=True)
    return {
        "kind": kind,
        "symbol": symbol,
        "timeframe": timeframe,
        "exists": True,
        "rows": int(len(frame)),
        "start_ts": ts.iloc[0].isoformat(),
        "end_ts": ts.iloc[-1].isoformat(),
        "path": str(path),
    }
