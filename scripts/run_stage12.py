"""Run Stage-12 full price-family completion and robustness sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage12.sweep import run_stage12_sweep, validate_stage12_summary_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-12 full price-family sweep")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated symbols override")
    parser.add_argument("--timeframes", type=str, default=None, help="Optional comma-separated timeframe override")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else None
    timeframes = [item.strip().lower() for item in str(args.timeframes).split(",") if item.strip()] if args.timeframes else None

    result = run_stage12_sweep(
        config=config,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframes=timeframes,
        runs_root=args.runs_dir,
        docs_dir=Path("docs"),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    summary = dict(result["summary"])
    forensic = dict(result.get("forensic_summary", {}))
    validate_stage12_summary_schema(summary)

    report_json = Path("docs") / "stage12_report_summary.json"
    if report_json.exists():
        payload = json.loads(report_json.read_text(encoding="utf-8"))
        payload["pytest_pass_count"] = payload.get("pytest_pass_count")
        report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage12_run_id: {result['run_id']}")
    print(f"total_combinations: {summary['total_combinations']}")
    print(f"valid_combinations: {summary['valid_combinations']}")
    print(f"runtime_seconds: {float(summary['runtime_seconds']):.4f}")
    print(f"verdict: {summary['verdict']}")
    if forensic:
        print(f"avg_backtest_ms_per_combo: {float(forensic.get('avg_backtest_ms_per_combo', 0.0)):.6f}")
        print(f"zero_trade_pct: {float(forensic.get('zero_trade_pct', 0.0)):.6f}")
        print(f"walkforward_executed_true_pct: {float(forensic.get('walkforward_executed_true_pct', 0.0)):.6f}")
        print(f"mc_trigger_rate: {float(forensic.get('mc_trigger_rate', 0.0)):.6f}")
        print(f"stage12_1_classification: {forensic.get('final_stage12_1_classification', '')}")
    top_rows = summary.get("top_robust", [])[:3]
    for idx, row in enumerate(top_rows, start=1):
        print(
            f"top_{idx}: {row.get('symbol')} | {row.get('timeframe')} | {row.get('strategy')} | "
            f"{row.get('exit_type')} | robust_score={float(row.get('robust_score', 0.0)):.6f}"
        )
    print("wrote: docs/stage12_report.md")
    print("wrote: docs/stage12_report_summary.json")
    print("wrote: docs/stage12_1_execution_forensics_report.md")
    print("wrote: docs/stage12_1_execution_forensics_summary.json")
    print(f"leaderboard: {result['run_dir'] / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
