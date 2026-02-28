"""Historical OHLCV loader via ccxt with deterministic normalization."""

from __future__ import annotations

from typing import Iterable

import ccxt
import pandas as pd

from buffmini.constants import OHLCV_COLUMNS
from buffmini.utils.time import parse_utc_timestamp


SUPPORTED_FETCH_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d")


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
    exchange: ccxt.Exchange | None = None,
) -> pd.DataFrame:
    """Fetch Binance OHLCV and return standardized deterministic DataFrame."""

    timeframe_value = str(timeframe).strip().lower()
    if timeframe_value not in SUPPORTED_FETCH_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Allowed: {SUPPORTED_FETCH_TIMEFRAMES}")
    if int(limit) <= 0:
        raise ValueError("limit must be >= 1")

    exchange_instance = exchange or ccxt.binance({"enableRateLimit": True})
    exchange_instance.load_markets()
    market_symbol = _resolve_market_symbol(exchange_instance, symbol)
    timeframe_ms = exchange_instance.parse_timeframe(timeframe_value) * 1000

    since_ts = parse_utc_timestamp(start)
    end_ts = parse_utc_timestamp(end)
    since_ms = int(since_ts.timestamp() * 1000) if since_ts is not None else None
    end_ms = int(end_ts.timestamp() * 1000) if end_ts is not None else None

    candles: list[list[float]] = []
    cursor = since_ms

    while True:
        batch: Iterable[list[float]] = exchange_instance.fetch_ohlcv(
            symbol=market_symbol,
            timeframe=timeframe_value,
            since=cursor,
            limit=int(limit),
        )
        batch = list(batch)
        if not batch:
            break

        for row in batch:
            ts = int(row[0])
            if end_ms is not None and ts > end_ms:
                continue
            candles.append(row)

        next_cursor = int(batch[-1][0]) + timeframe_ms
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if end_ms is not None and cursor > end_ms:
            break
        if len(batch) < int(limit):
            break

    if not candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    return standardize_ohlcv_frame(pd.DataFrame(candles, columns=OHLCV_COLUMNS))


def standardize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV schema/dtypes, enforce UTC timestamps and monotonic order."""

    if frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    required = set(OHLCV_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    df = frame.loc[:, OHLCV_COLUMNS].copy()
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return df


def validate_ohlcv_frame(frame: pd.DataFrame) -> None:
    """Validate deterministic OHLCV frame contract."""

    required = set(OHLCV_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if frame.empty:
        return

    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("timestamp contains invalid values")
    if bool(ts.duplicated().any()):
        raise ValueError("timestamp contains duplicates")
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamp must be monotonic increasing")

    for column in ["open", "high", "low", "close", "volume"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} contains non-numeric values")


def merge_ohlcv_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append and deduplicate rows by timestamp with stable ordering."""

    left = standardize_ohlcv_frame(existing if isinstance(existing, pd.DataFrame) else pd.DataFrame(columns=OHLCV_COLUMNS))
    right = standardize_ohlcv_frame(incoming if isinstance(incoming, pd.DataFrame) else pd.DataFrame(columns=OHLCV_COLUMNS))
    if left.empty:
        merged = right
    elif right.empty:
        merged = left
    else:
        merged = pd.concat([left, right], ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    validate_ohlcv_frame(merged)
    return merged


def _resolve_market_symbol(exchange: ccxt.Exchange, symbol: str) -> str:
    if symbol in exchange.markets:
        return symbol
    futures_symbol = f"{symbol}:USDT"
    if futures_symbol in exchange.markets:
        return futures_symbol
    if symbol.endswith(":USDT") and symbol in exchange.markets:
        return symbol
    raise ValueError(f"Symbol not available on exchange: {symbol}")

