"""Download/update OHLCV data for configured universe."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.data.loader import fetch_ohlcv
from buffmini.data.storage import save_parquet
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update 1h OHLCV parquet data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

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

        path = save_parquet(frame=frame, symbol=symbol, timeframe=timeframe)
        logger.info("Saved %s rows to %s", len(frame), path)


if __name__ == "__main__":
    main()
