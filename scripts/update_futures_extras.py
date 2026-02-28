"""Download funding/open-interest futures extras and persist under data/derived."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.derived_store import load_derived_parquet, save_derived_parquet, write_meta_json
from buffmini.data.futures_extras import (
    align_funding_to_ohlcv,
    align_open_interest_to_ohlcv,
    create_binance_futures_exchange,
    fetch_funding_history,
    fetch_open_interest_history_backfill,
    funding_quality_report,
    open_interest_coverage_report,
    open_interest_quality_report,
)
from buffmini.data.store import build_data_store
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Stage-9 futures extras")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=Path("data") / "derived")
    parser.add_argument("--force-backfill", action="store_true", help="Force full OI historical backfill attempt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config.get("data", {})
    extras_cfg = data_cfg.get("futures_extras", {}) if isinstance(data_cfg, dict) else {}
    symbols = list(extras_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"]))
    timeframe = str(extras_cfg.get("timeframe", config["universe"]["timeframe"]))
    if timeframe != "1h":
        raise ValueError("Stage-9 futures extras support only 1h timeframe")

    store = build_data_store(
        backend=str(config.get("data", {}).get("backend", "parquet")),
        data_dir=args.data_dir,
        base_timeframe=str(config.get("universe", {}).get("base_timeframe") or timeframe),
        resample_source=str(config.get("data", {}).get("resample_source", "direct")),
        derived_dir=args.derived_dir,
        partial_last_bucket=bool(config.get("data", {}).get("partial_last_bucket", False)),
    )
    exchange = create_binance_futures_exchange()

    for symbol in symbols:
        ohlcv = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if ohlcv.empty:
            print(f"skip {symbol}: no OHLCV rows")
            continue

        start_ms = int(ohlcv["timestamp"].iloc[0].timestamp() * 1000)
        end_ms = int(ohlcv["timestamp"].iloc[-1].timestamp() * 1000)

        funding_raw = fetch_funding_history(exchange=exchange, symbol=symbol, start_ms=start_ms, end_ms=end_ms)
        funding_aligned = align_funding_to_ohlcv(ohlcv=ohlcv, funding=funding_raw, timeframe=timeframe)

        save_derived_parquet(
            frame=funding_aligned,
            kind="funding",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        quality = funding_quality_report(funding_raw)
        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "binance",
            "kind": "funding",
            "start_ts": quality["start_ts"],
            "end_ts": quality["end_ts"],
            "row_count": int(quality["rows"]),
            "gaps_count": int(quality["gaps_count"]),
            "fetched_at_utc": utc_now_compact(),
        }
        write_meta_json(
            kind="funding",
            symbol=symbol,
            timeframe=timeframe,
            payload=meta,
            data_dir=args.derived_dir,
        )

        existing_oi = _load_existing_oi_aligned(
            symbol=symbol,
            timeframe=timeframe,
            derived_dir=args.derived_dir,
        )
        existing_oi_non_null = int(existing_oi["open_interest"].notna().sum()) if not existing_oi.empty else 0
        need_backfill = bool(args.force_backfill) or existing_oi_non_null == 0
        if not need_backfill and not existing_oi.empty:
            earliest_existing = existing_oi.loc[existing_oi["open_interest"].notna(), "timestamp"]
            if not earliest_existing.empty:
                need_backfill = bool(earliest_existing.min() > ohlcv["timestamp"].min())
            else:
                need_backfill = True
        fetch_start_ms = start_ms
        if not need_backfill:
            recent_existing = existing_oi.loc[existing_oi["open_interest"].notna(), "timestamp"]
            if not recent_existing.empty:
                overlap_buffer_ms = 7 * 24 * 3600 * 1000
                fetch_start_ms = max(start_ms, int(recent_existing.max().timestamp() * 1000) - overlap_buffer_ms)

        oi_raw, oi_fetch_info = fetch_open_interest_history_backfill(
            exchange=exchange,
            symbol=symbol,
            start_ms=fetch_start_ms,
            end_ms=end_ms,
            timeframe=timeframe,
            limit=500,
            max_retries=3,
            retry_backoff_sec=0.8,
            sleep_between_chunks_sec=0.0,
        )
        oi_aligned = align_open_interest_to_ohlcv(ohlcv=ohlcv, open_interest=oi_raw, timeframe=timeframe)
        oi_aligned = _merge_aligned_series(
            existing=existing_oi,
            incoming=oi_aligned,
            value_col="open_interest",
        )

        save_derived_parquet(
            frame=oi_aligned,
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        oi_quality = open_interest_quality_report(oi_raw)
        oi_coverage = open_interest_coverage_report(
            open_interest=oi_raw,
            expected_start_ts=ohlcv["timestamp"].min(),
            expected_end_ts=ohlcv["timestamp"].max(),
            timeframe=timeframe,
        )
        previous_oi_meta = _read_existing_meta(
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            derived_dir=args.derived_dir,
        )
        warnings = list(oi_coverage.get("warnings", []))
        warnings.extend([str(item) for item in oi_fetch_info.get("warnings", [])])
        oi_meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": "binance",
            "kind": "open_interest",
            "start_ts": oi_coverage["start_ts"],
            "end_ts": oi_coverage["end_ts"],
            "row_count": int(oi_coverage["row_count"]),
            "total_expected_rows": int(oi_coverage["total_expected_rows"]),
            "coverage_ratio": float(oi_coverage["coverage_ratio"]),
            "gaps_count": int(oi_coverage["gap_count"]),
            "largest_gap_hours": float(oi_coverage["largest_gap_hours"]),
            "quality_rows": int(oi_quality["rows"]),
            "quality_gaps_count": int(oi_quality["gaps_count"]),
            "requests_count": int(oi_fetch_info.get("requests_count", 0)),
            "stop_reason": str(oi_fetch_info.get("stop_reason", "unknown")),
            "backfill_attempted": bool(need_backfill),
            "force_backfill": bool(args.force_backfill),
            "fetched_at_utc": utc_now_compact(),
            "warnings": sorted({str(w) for w in warnings if str(w).strip()}),
            "previous_start_ts": previous_oi_meta.get("start_ts"),
            "previous_end_ts": previous_oi_meta.get("end_ts"),
            "previous_row_count": previous_oi_meta.get("row_count"),
            "previous_coverage_ratio": previous_oi_meta.get("coverage_ratio"),
        }
        write_meta_json(
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            payload=oi_meta,
            data_dir=args.derived_dir,
        )

        print(
            f"funding {symbol} | rows={meta['row_count']} | range={meta['start_ts']}..{meta['end_ts']} | "
            f"gaps={meta['gaps_count']}"
        )
        print(
            f"open_interest {symbol} | rows={oi_meta['row_count']}/{oi_meta['total_expected_rows']} "
            f"(coverage={oi_meta['coverage_ratio']:.4f}) | range={oi_meta['start_ts']}..{oi_meta['end_ts']} | "
            f"gaps={oi_meta['gaps_count']} | stop={oi_meta['stop_reason']}"
        )
        if oi_meta["warnings"]:
            print(f"open_interest {symbol} warnings: {oi_meta['warnings']}")


def _load_existing_oi_aligned(symbol: str, timeframe: str, derived_dir: Path):
    try:
        frame = load_derived_parquet(
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=derived_dir,
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame[["timestamp", "open_interest"]].sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )


def _merge_aligned_series(existing, incoming, value_col: str):
    if existing is None or existing.empty:
        base = incoming.copy()
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        return base.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if incoming is None or incoming.empty:
        base = existing.copy()
        base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
        return base.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    left = existing.copy()
    right = incoming.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)

    left = left.set_index("timestamp")
    right = right.set_index("timestamp")
    merged = right.combine_first(left)
    merged = merged[[value_col]].reset_index()
    return merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def _read_existing_meta(kind: str, symbol: str, timeframe: str, derived_dir: Path) -> dict:
    from buffmini.data.derived_store import read_meta_json

    try:
        payload = read_meta_json(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=derived_dir)
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
