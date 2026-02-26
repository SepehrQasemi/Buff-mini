"""Parquet persistence for market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.constants import RAW_DATA_DIR


def save_parquet(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    data_dir: str | Path = RAW_DATA_DIR,
) -> Path:
    """Save OHLCV frame to parquet under data/raw."""

    path = parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_parquet(
    symbol: str,
    timeframe: str,
    data_dir: str | Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load OHLCV parquet from data/raw."""

    path = parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if not path.exists():
        msg = f"Parquet not found: {path}"
        raise FileNotFoundError(msg)

    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def parquet_path(symbol: str, timeframe: str, data_dir: str | Path = RAW_DATA_DIR) -> Path:
    """Return standardized parquet path for a symbol/timeframe pair."""

    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return Path(data_dir) / f"{safe_symbol}_{timeframe}.parquet"
