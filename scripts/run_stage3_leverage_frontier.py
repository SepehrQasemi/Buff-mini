"""Run Stage-3.2 leverage frontier on Stage-2 portfolio methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.leverage_frontier import run_stage3_leverage_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-3.2 leverage frontier")
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leverage-levels", type=str, default="1,2,3,5,10,15,20")
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--bootstrap", type=str, default="block")
    parser.add_argument("--block-size-trades", type=int, default=10)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--ruin-dd-threshold", type=float, default=0.5)
    parser.add_argument("--max-p-ruin", type=float, default=0.01)
    parser.add_argument("--max-dd-p95", type=float, default=0.25)
    parser.add_argument("--min-return-p05", type=float, default=0.0)
    parser.add_argument("--methods", type=str, default="equal,vol,corr-min")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leverage_levels = [float(item.strip()) for item in str(args.leverage_levels).split(",") if item.strip()]
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]

    run_dir = run_stage3_leverage_frontier(
        stage2_run_id=args.stage2_run_id,
        leverage_levels=leverage_levels,
        n_paths=args.n_paths,
        bootstrap=args.bootstrap,
        block_size_trades=args.block_size_trades,
        initial_equity=args.initial_equity,
        ruin_dd_threshold=args.ruin_dd_threshold,
        max_p_ruin=args.max_p_ruin,
        max_dd_p95=args.max_dd_p95,
        min_return_p05=args.min_return_p05,
        methods=methods,
        seed=args.seed,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
    )

    summary_path = run_dir / "stage3_2_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"stage3_2 run_id: {summary['run_id']}")
    for method_key, payload in summary["methods"].items():
        print(
            f"{method_key}: chosen_safe_leverage={payload['chosen_safe_leverage']}, "
            f"first_failure_leverage={payload['first_failure_leverage']}, "
            f"first_failure_constraints={payload['first_failure_constraints']}"
        )
    print(f"artifacts: {run_dir}")


if __name__ == "__main__":
    main()

