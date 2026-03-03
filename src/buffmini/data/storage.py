"""Parquet persistence for market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, RAW_DATA_DIR
from buffmini.data.timeframe_files import timeframe_to_file_token


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

    path = resolve_parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    if path is None:
        fallback = parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
        msg = f"Parquet not found: {fallback}"
        raise FileNotFoundError(msg)

    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def parquet_path(symbol: str, timeframe: str, data_dir: str | Path = RAW_DATA_DIR) -> Path:
    """Return standardized parquet path for a symbol/timeframe pair."""

    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return Path(data_dir) / f"{safe_symbol}_{timeframe}.parquet"


def nested_parquet_path(
    symbol: str,
    timeframe: str,
    *,
    data_dir: str | Path = RAW_DATA_DIR,
    exchange: str = "binance",
) -> Path:
    """Return nested parquet path under data/raw/<exchange>/<symbol>/<tf>.parquet."""

    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return Path(data_dir) / str(exchange).strip().lower() / safe_symbol / f"{timeframe}.parquet"


def canonical_parquet_path(
    symbol: str,
    timeframe: str,
    *,
    canonical_dir: str | Path = CANONICAL_DATA_DIR,
    exchange: str = "binance",
) -> Path:
    """Return canonical parquet path under data/canonical/<exchange>/<symbol>/<tf>.parquet."""

    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    token = timeframe_to_file_token(str(timeframe))
    return Path(canonical_dir) / str(exchange).strip().lower() / safe_symbol / f"{token}.parquet"


def resolve_parquet_path(symbol: str, timeframe: str, data_dir: str | Path = RAW_DATA_DIR) -> Path | None:
    """Resolve best available path across canonical, nested raw, and legacy flat raw."""

    raw_dir = Path(data_dir).resolve()
    default_raw_dir = Path(RAW_DATA_DIR).resolve()
    candidates: list[Path] = []
    # Only probe global canonical store when using the default raw location.
    if raw_dir == default_raw_dir:
        candidates.append(canonical_parquet_path(symbol=symbol, timeframe=timeframe))
        # Backward compatibility for pre-tokenized canonical files.
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        candidates.append(Path(CANONICAL_DATA_DIR) / "binance" / safe_symbol / f"{timeframe}.parquet")
    candidates.extend(
        [
            nested_parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir),
            parquet_path(symbol=symbol, timeframe=timeframe, data_dir=data_dir),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
