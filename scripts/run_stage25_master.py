"""Run unified Stage-25 master program (research + live + replay)."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage25.edge_program import run_stage25_master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-25 master program")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--families", type=str, default="price,volatility,flow")
    parser.add_argument("--composers", type=str, default="weighted_sum")
    parser.add_argument("--cost-levels", type=str, default="realistic,high")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = run_stage25_master(
        config=cfg,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=_csv(args.symbols),
        timeframes=_csv(args.timeframes),
        families=_csv(args.families),
        composers=_csv(args.composers),
        cost_levels=_csv(args.cost_levels),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        docs_dir=args.docs_dir,
    )
    summary = dict(result["summary"])
    print(f"research_run_id: {summary['research_run_id']}")
    print(f"live_run_id: {summary['live_run_id']}")
    print(f"regime_run_id: {summary['regime_run_id']}")
    print(f"final_verdict: {summary['final_verdict']}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")


if __name__ == "__main__":
    main()

