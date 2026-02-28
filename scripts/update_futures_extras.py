"""Download funding/open-interest futures extras and persist under data/derived."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.derived_store import save_derived_parquet, write_meta_json
from buffmini.data.futures_extras import (
    align_funding_to_ohlcv,
    create_binance_futures_exchange,
    fetch_funding_history,
    funding_quality_report,
)
from buffmini.data.store import build_data_store
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Stage-9 futures extras")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=Path("data") / "derived")
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

        print(
            f"funding {symbol} | rows={meta['row_count']} | range={meta['start_ts']}..{meta['end_ts']} | "
            f"gaps={meta['gaps_count']}"
        )


if __name__ == "__main__":
    main()
