"""Run Stage-10 baseline vs upgraded evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import run_stage10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-10 evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic data")
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-mode", type=str, default="v2", choices=["simple", "v2"])
    parser.add_argument("--walkforward-v2", dest="walkforward_v2", action="store_true", default=True)
    parser.add_argument("--no-walkforward-v2", dest="walkforward_v2", action="store_false")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else None
    summary = run_stage10(
        config=config,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframe=str(args.timeframe),
        cost_mode=str(args.cost_mode),
        walkforward_v2_enabled=bool(args.walkforward_v2),
        runs_root=args.runs_dir,
        docs_dir=Path("docs"),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    print(f"run_id: {summary['run_id']}")
    print("wrote: docs/stage10_report.md")
    print("wrote: docs/stage10_report_summary.json")


if __name__ == "__main__":
    main()
