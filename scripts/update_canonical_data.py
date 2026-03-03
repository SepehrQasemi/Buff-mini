"""Stage-26.9.1: resume-safe 1m downloader into canonical raw layout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.canonical_raw import detect_gaps_minutes, file_sha256, prepare_frame, raw_meta_path, raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update canonical raw OHLCV data (resume-safe)")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-batches", type=int, default=0, help="Safety cap for one invocation (0=unbounded)")
    parser.add_argument("--max-retries", type=int, default=6, help="Retries per fetch_ohlcv request")
    parser.add_argument("--retry-sleep-ms", type=int, default=1000, help="Base retry backoff in milliseconds")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--compression", type=str, default="zstd")
    return parser.parse_args()


_RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
)


def _fetch_ohlcv_retry(
    *,
    exchange: ccxt.Exchange,
    market_symbol: str,
    timeframe: str,
    since: int,
    limit: int,
    max_retries: int,
    retry_sleep_ms: int,
) -> list[list[float]]:
    for attempt in range(int(max_retries) + 1):
        try:
            return list(exchange.fetch_ohlcv(symbol=market_symbol, timeframe=timeframe, since=since, limit=limit))
        except _RETRYABLE_EXCEPTIONS:
            if attempt >= int(max_retries):
                raise
            base_sleep = max(float(retry_sleep_ms), float(getattr(exchange, "rateLimit", 0.0))) / 1000.0
            time.sleep(base_sleep * (2.0**attempt))


def _fetch_incremental(
    *,
    exchange: ccxt.Exchange,
    market_symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    max_batches: int,
    max_retries: int,
    retry_sleep_ms: int,
) -> pd.DataFrame:
    timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    cursor = int(start_ms)
    out: list[list[float]] = []
    batch_count = 0
    while cursor <= int(end_ms):
        batch = _fetch_ohlcv_retry(
            exchange=exchange,
            market_symbol=market_symbol,
            timeframe=timeframe,
            since=cursor,
            limit=int(limit),
            max_retries=int(max_retries),
            retry_sleep_ms=int(retry_sleep_ms),
        )
        batch_count += 1
        if not batch:
            break
        for row in batch:
            ts = int(row[0])
            if ts > int(end_ms):
                continue
            out.append(row)
        next_cursor = int(batch[-1][0]) + timeframe_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < int(limit):
            break
        if int(max_batches) > 0 and batch_count >= int(max_batches):
            break
        sleep_seconds = max(float(getattr(exchange, "rateLimit", 0.0)) / 1000.0, 0.0)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    if not out:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return prepare_frame(pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"]))


def main() -> None:
    args = parse_args()
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    if not symbols:
        raise ValueError("No symbols provided")
    timeframe = str(args.timeframe).strip()
    if timeframe != "1m":
        raise ValueError("Stage-26.9.1 downloader currently supports timeframe=1m only")

    exchange_name = str(args.exchange).strip().lower()
    if exchange_name != "binance":
        raise ValueError("Only binance is supported in this downloader")
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()

    now = pd.Timestamp.now(tz="UTC")
    end_ms = int(now.timestamp() * 1000)
    start_ts = now - pd.Timedelta(days=float(args.years) * 365.25)
    start_ms = int(start_ts.timestamp() * 1000)

    for symbol in symbols:
        market_symbol = symbol if symbol in exchange.markets else f"{symbol}:USDT"
        if market_symbol not in exchange.markets:
            raise ValueError(f"Symbol not available: {symbol}")

        out_path = raw_path(data_dir=args.data_dir, exchange=exchange_name, symbol=symbol, timeframe=timeframe)
        out_meta = raw_meta_path(data_dir=args.data_dir, exchange=exchange_name, symbol=symbol, timeframe=timeframe)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if out_path.exists():
            existing = prepare_frame(pd.read_parquet(out_path))
        since_ms = start_ms
        until_head_ms = None
        if not existing.empty:
            existing_ts = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce").dropna()
            first = int(existing_ts.iloc[0].timestamp() * 1000)
            last = int(existing_ts.iloc[-1].timestamp() * 1000)
            if first > start_ms:
                until_head_ms = first - 60_000
            since_ms = max(start_ms, last + 60_000)

        head = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if until_head_ms is not None and until_head_ms >= start_ms:
            head = _fetch_incremental(
                exchange=exchange,
                market_symbol=market_symbol,
                timeframe=timeframe,
                start_ms=start_ms,
                end_ms=until_head_ms,
                limit=int(args.limit),
                max_batches=int(args.max_batches),
                max_retries=int(args.max_retries),
                retry_sleep_ms=int(args.retry_sleep_ms),
            )

        tail = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if since_ms <= end_ms:
            tail = _fetch_incremental(
                exchange=exchange,
                market_symbol=market_symbol,
                timeframe=timeframe,
                start_ms=since_ms,
                end_ms=end_ms,
                limit=int(args.limit),
                max_batches=int(args.max_batches),
                max_retries=int(args.max_retries),
                retry_sleep_ms=int(args.retry_sleep_ms),
            )
        merged = prepare_frame(pd.concat([head, existing, tail], ignore_index=True))
        merged.to_parquet(out_path, index=False, compression=str(args.compression))
        sha = file_sha256(out_path)
        gaps = detect_gaps_minutes(merged["timestamp"], expected_minutes=1)
        ts = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce").dropna()
        payload: dict[str, Any] = {
            "exchange": exchange_name,
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "requested_years": int(args.years),
            "actual_start_ts": ts.iloc[0].isoformat() if not ts.empty else None,
            "actual_end_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
            "candle_count": int(merged.shape[0]),
            "sha256": sha,
            "gaps_detected": {"count": int(gaps.gaps_detected), "max_gap_minutes": int(gaps.max_gap_minutes)},
            "generator_version": "stage26_9",
        }
        out_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"{symbol} {timeframe}: rows={merged.shape[0]} path={out_path}")


if __name__ == "__main__":
    main()
