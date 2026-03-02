"""Run Stage-24 capital-level simulations across initial equity levels."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage24.capital_sim import run_stage24_capital_sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-24 capital-level simulations")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--operational-timeframe", type=str, default="1h")
    parser.add_argument("--mode", type=str, choices=["risk_pct", "alloc_pct"], default="risk_pct")
    parser.add_argument("--initial-equities", type=str, default="100,1000,10000,100000")
    parser.add_argument("--out-run-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _floats(value: str) -> list[float]:
    out: list[float] = []
    for item in _csv(value):
        out.append(float(item))
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dry_run = bool(args.dry_run)
    if bool(args.use_real_data):
        dry_run = False
    result = run_stage24_capital_sim(
        config=cfg,
        seed=int(args.seed),
        dry_run=dry_run,
        symbols=_csv(args.symbols),
        base_timeframe=str(args.base_timeframe),
        operational_timeframe=str(args.operational_timeframe),
        mode=str(args.mode),
        initial_equities=_floats(args.initial_equities),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        out_run_id=args.out_run_id,
        docs_dir=args.docs_dir,
    )
    summary = dict(result["summary"])
    print(f"run_id: {summary['run_id']}")
    print(f"results_csv: {result['results_csv']}")
    print(f"results_json: {result['results_json']}")
    print(f"capital_doc: {result['capital_doc']}")
    print(f"results_hash: {summary['results_hash']}")


if __name__ == "__main__":
    main()
