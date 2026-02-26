"""Download/update OHLCV data for configured universe."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.loader import fetch_ohlcv
from buffmini.data.store import build_data_store
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update 1h OHLCV parquet data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    store = build_data_store(
        backend=str(config.get("data", {}).get("backend", "parquet")),
        data_dir=args.data_dir,
    )

    universe = config["universe"]
    timeframe = universe["timeframe"]
    start = universe["start"]
    end = universe["end"]

    for symbol in universe["symbols"]:
        logger.info("Fetching %s %s", symbol, timeframe)
        frame = fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
        if frame.empty:
            logger.warning("No rows fetched for %s", symbol)
            continue

        store.save_ohlcv(symbol=symbol, timeframe=timeframe, df=frame)
        logger.info("Saved %s rows for %s %s", len(frame), symbol, timeframe)


if __name__ == "__main__":
    main()
