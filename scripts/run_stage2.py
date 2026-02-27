"""Run Stage-2 portfolio construction from an existing Stage-1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.builder import run_stage2_portfolio
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-2 portfolio evaluation")
    parser.add_argument("--stage1-run-id", type=str, required=True)
    parser.add_argument("--method", type=str, default="all")
    parser.add_argument("--forward-days", type=int, default=30)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_stage2_portfolio(
        stage1_run_id=args.stage1_run_id,
        method=args.method,
        forward_days=args.forward_days,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
    )

    summary_path = run_dir / "portfolio_summary.json"
    if not summary_path.exists():
        logger.warning("Stage-2 completed but portfolio_summary.json not found at %s", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("Stage-2 Portfolio Summary")
    print(f"run_id: {summary['run_id']}")
    print(f"average_correlation: {summary['average_correlation']:.4f}")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["portfolio_methods"].get(method_key)
        if payload is None:
            continue
        holdout = payload["holdout"]
        forward = payload["forward"]
        print(
            f"{method_key}: holdout_pf={float(holdout['profit_factor']):.4f}, "
            f"forward_pf={float(forward['profit_factor']):.4f}, "
            f"holdout_max_dd={float(holdout['max_drawdown']):.4f}, "
            f"forward_max_dd={float(forward['max_drawdown']):.4f}, "
            f"effective_n={float(payload['effective_number_of_strategies']):.4f}"
        )


if __name__ == "__main__":
    main()
