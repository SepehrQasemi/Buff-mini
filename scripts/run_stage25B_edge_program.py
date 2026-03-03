"""Run Stage-25B edge program in research/live mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage25.edge_program import run_stage25b_edge_program
from buffmini.stage27.coverage_gate import evaluate_coverage_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-25B family edge program")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--mode", type=str, choices=["research", "live"], default="research")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--families", type=str, default="price,volatility,flow")
    parser.add_argument("--composers", type=str, default="weighted_sum")
    parser.add_argument("--cost-levels", type=str, default="realistic,high")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--out-run-id", type=str, default=None)
    parser.add_argument("--allow-insufficient-data", action="store_true")
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    requested_symbols = _csv(args.symbols)
    base_tf = str((cfg.get("evaluation", {}) or {}).get("stage26", {}).get("base_timeframe", "1m"))
    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=requested_symbols,
        timeframe=base_tf,
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )
    if not gate.can_run:
        print(f"coverage_gate_status: {gate.status}")
        print(f"coverage_years_by_symbol: {gate.coverage_years_by_symbol}")
        raise SystemExit(2)
    if gate.disabled_symbols:
        print(f"auto_disabled_symbols: {','.join(gate.disabled_symbols)}")
    result = run_stage25b_edge_program(
        config=cfg,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        mode=str(args.mode),
        symbols=list(gate.used_symbols),
        timeframes=_csv(args.timeframes),
        families=_csv(args.families),
        composers=_csv(args.composers),
        cost_levels=_csv(args.cost_levels),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        docs_dir=args.docs_dir,
        out_run_id=args.out_run_id,
    )
    summary = dict(result["summary"])
    print(f"run_id: {summary['run_id']}")
    print(f"mode: {summary['mode']}")
    print(f"status: {summary['status']}")
    print(f"results_csv: {result['results_csv']}")
    print(f"results_json: {result['results_json']}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")


if __name__ == "__main__":
    main()
