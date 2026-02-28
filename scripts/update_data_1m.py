"""Convenience wrapper to update 1m OHLCV data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update 1m OHLCV parquet data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force-backfill", action="store_true")
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd: list[str] = [
        sys.executable,
        "scripts/update_data.py",
        "--config",
        str(args.config),
        "--data-dir",
        str(args.data_dir),
        "--timeframe",
        "1m",
        "--limit",
        str(int(args.limit)),
    ]
    if args.symbols:
        cmd.extend(["--symbols", str(args.symbols)])
    if args.start:
        cmd.extend(["--start", str(args.start)])
    if args.end:
        cmd.extend(["--end", str(args.end)])
    if bool(args.force_backfill):
        cmd.append("--force-backfill")
    subprocess.run(cmd, check=True, shell=False)


if __name__ == "__main__":
    main()

