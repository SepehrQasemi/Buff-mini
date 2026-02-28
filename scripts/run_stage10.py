"""Run Stage-10 baseline vs upgraded evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import (
    build_stage10_6_report_from_runs,
    build_stage10_7_report_from_runs,
    run_stage10,
    run_stage10_exit_ab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-10 evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic data")
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-mode", type=str, default="v2", choices=["simple", "v2"])
    parser.add_argument(
        "--exit-mode",
        type=str,
        default="single",
        choices=["single", "compare"],
        help="single: run one exit mode; compare: run fixed_atr vs atr_trailing A/B",
    )
    parser.add_argument(
        "--exit",
        type=str,
        default=None,
        choices=["fixed_atr", "atr_trailing", "breakeven_1r", "partial_tp", "regime_flip_exit"],
        help="Optional single exit mode for Stage-10 A/B isolation runs",
    )
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
    if str(args.exit_mode) == "compare":
        compare = run_stage10_exit_ab(
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
        summary = dict(compare["selected_summary"])
        print(f"exit_ab_run_id: {compare['run_id']}")
        print(f"selected_exit: {compare['selected_exit']}")
        print("wrote: runs/<exit_ab_run_id>/exit_ab_compare.csv")
    else:
        summary = run_stage10(
            config=config,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            cost_mode=str(args.cost_mode),
            walkforward_v2_enabled=bool(args.walkforward_v2),
            exit_mode=str(args.exit) if args.exit else None,
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
    stage10_6 = build_stage10_6_report_from_runs(
        runs_root=args.runs_dir,
        docs_dir=Path("docs"),
        max_drop_pct=10.0,
    )
    stage10_7 = build_stage10_7_report_from_runs(
        runs_root=args.runs_dir,
        docs_dir=Path("docs"),
        max_drop_pct=10.0,
    )
    print(f"run_id: {summary['run_id']}")
    print("wrote: docs/stage10_report.md")
    print("wrote: docs/stage10_report_summary.json")
    print(f"stage10_6_report: docs/stage10_6_report.md (guard_pass={stage10_6['trade_count_guard']['pass']})")
    print(f"stage10_7_report: docs/stage10_7_report.md (verdict={stage10_7['final_verdict']})")


if __name__ == "__main__":
    main()
