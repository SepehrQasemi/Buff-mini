"""Stage-26.9.2 canonical timeframe builder from canonical raw 1m."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, RAW_DATA_DIR
from buffmini.data.canonical_raw import file_sha256, prepare_frame, raw_path, symbol_safe
from buffmini.data.loader import validate_ohlcv_frame
from buffmini.data.resample import resample_monthly_ohlcv, resample_ohlcv

DEFAULT_CANONICAL_TFS = "1m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical timeframes deterministically from raw 1m")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--base", type=str, default="1m")
    parser.add_argument("--timeframes", type=str, default=DEFAULT_CANONICAL_TFS)
    parser.add_argument("--drop-incomplete-last", action="store_true", default=True)
    parser.add_argument("--keep-incomplete-last", action="store_true", default=False)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_DATA_DIR)
    parser.add_argument("--compression", type=str, default="zstd")
    return parser.parse_args()


def canonical_path(*, canonical_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return Path(canonical_dir) / str(exchange).strip().lower() / symbol_safe(symbol) / f"{timeframe}.parquet"


def canonical_meta_path(*, canonical_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return canonical_path(canonical_dir=canonical_dir, exchange=exchange, symbol=symbol, timeframe=timeframe).with_suffix(
        ".meta.json"
    )


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _build_frame(base_frame: pd.DataFrame, *, base_tf: str, target_tf: str, drop_incomplete_last: bool) -> pd.DataFrame:
    if str(target_tf) == str(base_tf):
        out = base_frame.copy()
    elif str(target_tf) == "1M":
        out = resample_monthly_ohlcv(base_frame, partial_last_bucket=not bool(drop_incomplete_last))
    else:
        out = resample_ohlcv(
            base_frame,
            target_timeframe=str(target_tf),
            base_timeframe=str(base_tf),
            partial_last_bucket=not bool(drop_incomplete_last),
        )
    validate_ohlcv_frame(out)
    return out


def build_for_symbol(
    *,
    symbol: str,
    exchange: str,
    base_tf: str,
    target_tfs: list[str],
    raw_dir: Path,
    canonical_dir: Path,
    compression: str,
    drop_incomplete_last: bool,
) -> dict[str, dict[str, Any]]:
    raw_file = raw_path(data_dir=raw_dir, exchange=exchange, symbol=symbol, timeframe=base_tf)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw base file missing: {raw_file}")
    base_frame = prepare_frame(pd.read_parquet(raw_file))
    source_hash = file_sha256(raw_file)
    out: dict[str, dict[str, Any]] = {}

    for tf in target_tfs:
        frame = _build_frame(base_frame, base_tf=base_tf, target_tf=tf, drop_incomplete_last=drop_incomplete_last)
        out_file = canonical_path(canonical_dir=canonical_dir, exchange=exchange, symbol=symbol, timeframe=tf)
        out_meta = canonical_meta_path(canonical_dir=canonical_dir, exchange=exchange, symbol=symbol, timeframe=tf)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_file, index=False, compression=str(compression))
        out_hash = file_sha256(out_file)
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        payload = {
            "timeframe": str(tf),
            "source_timeframe": str(base_tf),
            "exchange": str(exchange).strip().lower(),
            "symbol": str(symbol),
            "start_ts": ts.iloc[0].isoformat() if not ts.empty else None,
            "end_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
            "candle_count": int(frame.shape[0]),
            "sha256": str(out_hash),
            "source_sha256": str(source_hash),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "generator_version": "stage26_9",
            "drop_incomplete_last": bool(drop_incomplete_last),
            "compression": str(compression),
        }
        out_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        out[str(tf)] = payload
    return out


def main() -> None:
    args = parse_args()
    drop_incomplete_last = bool(args.drop_incomplete_last and not args.keep_incomplete_last)
    symbols = _parse_list(args.symbols)
    target_tfs = _parse_list(args.timeframes)
    if not symbols:
        raise ValueError("No symbols provided")
    if not target_tfs:
        raise ValueError("No timeframes provided")

    for symbol in symbols:
        result = build_for_symbol(
            symbol=symbol,
            exchange=str(args.exchange),
            base_tf=str(args.base),
            target_tfs=target_tfs,
            raw_dir=Path(args.raw_dir),
            canonical_dir=Path(args.canonical_dir),
            compression=str(args.compression),
            drop_incomplete_last=drop_incomplete_last,
        )
        print(f"{symbol}: built {len(result)} canonical timeframes")


if __name__ == "__main__":
    main()
