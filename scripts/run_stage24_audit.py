"""Run Stage-24 unified audit (baseline vs risk_pct vs alloc_pct + capital sim)."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage24.audit import run_stage24_audit
from buffmini.stage27.coverage_gate import evaluate_coverage_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-24 unified audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--operational-timeframe", type=str, default="1h")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--allow-insufficient-data", action="store_true")
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dry_run = bool(args.dry_run)
    if bool(args.use_real_data):
        dry_run = False
    requested_symbols = _csv(args.symbols)
    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=requested_symbols,
        timeframe=str(args.base_timeframe),
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

    equities = list(
        (cfg.get("evaluation", {}) or {})
        .get("stage24", {})
        .get("simulation", {})
        .get("initial_equities", [100.0, 1000.0, 10000.0, 100000.0])
    )
    result = run_stage24_audit(
        config=cfg,
        seed=int(args.seed),
        dry_run=dry_run,
        symbols=list(gate.used_symbols),
        base_timeframe=str(args.base_timeframe),
        operational_timeframe=str(args.operational_timeframe),
        initial_equities=[float(x) for x in equities],
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        docs_dir=args.docs_dir,
    )
    payload = dict(result["payload"])
    print(f"baseline_run_id: {result['baseline_run_id']}")
    print(f"risk_run_id: {result['risk_run_id']}")
    print(f"alloc_run_id: {result['alloc_run_id']}")
    print(f"capital_run_id: {result['capital_run_id']}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")
    print(f"verdict: {payload.get('verdict', '')}")
    print(
        "zero_trade_pct: "
        f"{float(payload.get('baseline', {}).get('zero_trade_pct', 0.0)):.6f} -> "
        f"{float(payload.get('risk_pct_mode', {}).get('zero_trade_pct', 0.0)):.6f}"
    )


if __name__ == "__main__":
    main()
