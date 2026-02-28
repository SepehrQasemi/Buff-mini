"""Run Stage-10.6 sandbox signal ranking."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.sandbox import run_stage10_sandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-10.6 sandbox ranking")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic data")
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-mode", type=str, default="v2", choices=["simple", "v2"])
    parser.add_argument("--exit", type=str, default="fixed_atr", choices=["fixed_atr", "atr_trailing"])
    parser.add_argument("--top-k-per-category", type=int, default=2)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else None
    summary = run_stage10_sandbox(
        config=config,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframe=str(args.timeframe),
        cost_mode=str(args.cost_mode),
        exit_mode=str(args.exit),
        top_k_per_category=int(args.top_k_per_category),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    print(f"run_id: {summary['run_id']}")
    print(f"enabled_signals: {summary['enabled_signals']}")
    print(f"rank_table: {summary['rank_table_path']}")


if __name__ == "__main__":
    main()
