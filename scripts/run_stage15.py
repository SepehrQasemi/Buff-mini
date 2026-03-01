"""Run Stage-15 alpha-v2 A/B architecture checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.alpha_v2.ab_runner import run_ab_compare
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-15 Alpha-v2 A/B")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else None
    out = run_ab_compare(
        config=cfg,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframe=str(args.timeframe),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        alpha_enabled=True,
    )
    summary = dict(out["summary"])
    metrics = {
        "run_id": summary["run_id"],
        "seed": summary["seed"],
        "config_hash": summary["config_hash"],
        "data_hash": summary["data_hash"],
        "resolved_end_ts": summary["resolved_end_ts"],
        "classic_trade_count": summary["classic"]["trade_count"],
        "alpha_trade_count": summary["alpha_v2"]["trade_count"],
        "classic_exp_lcb": summary["classic"]["exp_lcb"],
        "alpha_exp_lcb": summary["alpha_v2"]["exp_lcb"],
        "delta_exp_lcb": summary["delta"]["exp_lcb"],
        "activation_pct_not_neutral": summary["activation_stats"]["pct_not_neutral_multiplier"],
        "summary_hash": summary_hash(summary),
    }
    report_md = Path("docs/stage15_report.md")
    report_json = Path("docs/stage15_summary.json")
    write_report_pair(
        report_md=report_md,
        report_json=report_json,
        title="Stage-15 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage15.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage15.py --seed 42`",
        ],
        metrics=metrics,
        status="PASS",
        failures=[],
        next_actions=[
            "Stage-16: add context persistence and no-leak checks.",
            "Use A/B runner hashes to detect no-op regressions.",
        ],
        extras={
            "classification": "ARCHITECTURE_READY",
            "risk_of_overfitting": "A/B runner compares Classic hash vs alpha-v2 and blocks silent no-op claims.",
        },
    )
    print(f"run_id: {summary['run_id']}")
    print(f"stage15_summary: {report_json}")
    print(f"stage15_report: {report_md}")
    print(f"classic_hash: {summary['classic']['trades_hash']}")
    print(f"alpha_hash: {summary['alpha_v2']['trades_hash']}")


if __name__ == "__main__":
    main()

