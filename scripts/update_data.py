"""Download/update OHLCV parquet data for configured universe and timeframe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.loader import fetch_ohlcv, merge_ohlcv_frames, validate_ohlcv_frame
from buffmini.data.storage import load_parquet, parquet_path
from buffmini.data.store import build_data_store
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update OHLCV parquet data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols override.")
    parser.add_argument("--timeframe", type=str, default=None, help="Target download timeframe (1m/5m/15m/30m/1h/2h/4h/1d).")
    parser.add_argument("--base-timeframe", type=str, default=None, help="Alias for --timeframe.")
    parser.add_argument("--start", type=str, default=None, help="Optional UTC start timestamp.")
    parser.add_argument("--end", type=str, default=None, help="Optional UTC end timestamp.")
    parser.add_argument("--force-backfill", action="store_true", help="Ignore incremental cursor and fetch full start..end.")
    parser.add_argument("--limit", type=int, default=1000, help="ccxt fetch_ohlcv limit per request.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    store = build_data_store(
        backend=str(config.get("data", {}).get("backend", "parquet")),
        data_dir=args.data_dir,
    )

    universe = config["universe"]
    configured_symbols = list(universe.get("symbols", []))
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else configured_symbols
    if not symbols:
        raise ValueError("No symbols configured.")

    resolved_timeframe = str(
        args.base_timeframe
        or args.timeframe
        or universe.get("base_timeframe")
        or universe.get("timeframe")
        or "1h"
    ).strip().lower()

    configured_start = str(args.start or universe.get("start") or "2023-01-01T00:00:00Z")
    configured_end = args.end if args.end is not None else (universe.get("resolved_end_ts") or universe.get("end"))
    logger.info("Updating timeframe=%s symbols=%s", resolved_timeframe, ",".join(symbols))

    for symbol in symbols:
        path = parquet_path(symbol=symbol, timeframe=resolved_timeframe, data_dir=args.data_dir)
        existing = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if path.exists():
            existing = load_parquet(symbol=symbol, timeframe=resolved_timeframe, data_dir=args.data_dir)
            validate_ohlcv_frame(existing)

        fetch_start = configured_start
        if not bool(args.force_backfill) and not existing.empty:
            delta = _timeframe_to_timedelta(resolved_timeframe)
            last_ts = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce").dropna().iloc[-1]
            fetch_start = (last_ts + delta).isoformat()

        logger.info("Fetching %s %s start=%s end=%s", symbol, resolved_timeframe, fetch_start, configured_end)
        fetched = fetch_ohlcv(
            symbol=symbol,
            timeframe=resolved_timeframe,
            start=fetch_start,
            end=str(configured_end) if configured_end is not None else None,
            limit=int(args.limit),
        )
        merged = merge_ohlcv_frames(existing=existing, incoming=fetched)
        if merged.empty:
            logger.warning("No data for %s %s", symbol, resolved_timeframe)
            continue
        store.save_ohlcv(symbol=symbol, timeframe=resolved_timeframe, df=merged)
        _write_meta(path=path, frame=merged, symbol=symbol, timeframe=resolved_timeframe)
        logger.info("Saved %s rows for %s %s (delta=%s)", len(merged), symbol, resolved_timeframe, len(merged) - len(existing))


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip().lower()
    if text.endswith("m"):
        return pd.Timedelta(minutes=int(text[:-1]))
    if text.endswith("h"):
        return pd.Timedelta(hours=int(text[:-1]))
    if text.endswith("d"):
        return pd.Timedelta(days=int(text[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _write_meta(path: Path, frame: pd.DataFrame, symbol: str, timeframe: str) -> None:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    payload: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": str(timeframe),
        "source": "binance",
        "first_ts": ts.iloc[0].isoformat() if not ts.empty else None,
        "last_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
        "row_count": int(len(frame)),
        "fetched_at_utc": utc_now_compact(),
    }
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()

