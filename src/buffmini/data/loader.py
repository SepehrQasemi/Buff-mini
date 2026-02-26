"""Historical OHLCV loader via ccxt."""

from __future__ import annotations

from typing import Iterable

import ccxt
import pandas as pd

from buffmini.constants import OHLCV_COLUMNS
from buffmini.utils.time import parse_utc_timestamp


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch Binance OHLCV data and return standardized DataFrame."""

    exchange = ccxt.binance({"enableRateLimit": True})
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    since_ts = parse_utc_timestamp(start)
    end_ts = parse_utc_timestamp(end)
    since_ms = int(since_ts.timestamp() * 1000) if since_ts is not None else None
    end_ms = int(end_ts.timestamp() * 1000) if end_ts is not None else None

    candles: list[list[float]] = []
    cursor = since_ms

    while True:
        batch: Iterable[list[float]] = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=cursor,
            limit=limit,
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
        if len(batch) < limit:
            break

    if not candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df
