"""Run Stage-3.1 Monte Carlo robustness simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.monte_carlo import run_stage3_monte_carlo
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-3.1 Monte Carlo robustness simulation")
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--methods", type=str, default="equal,vol,corr-min")
    parser.add_argument("--bootstrap", type=str, default="block")
    parser.add_argument("--block-size-trades", type=int, default=10)
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--ruin-dd-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--save-paths", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    run_dir = run_stage3_monte_carlo(
        stage2_run_id=args.stage2_run_id,
        methods=methods,
        bootstrap=args.bootstrap,
        block_size_trades=args.block_size_trades,
        n_paths=args.n_paths,
        initial_equity=args.initial_equity,
        ruin_dd_threshold=args.ruin_dd_threshold,
        seed=args.seed,
        leverage=args.leverage,
        save_paths=args.save_paths,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
    )

    summary_path = run_dir / "mc_summary.json"
    if not summary_path.exists():
        logger.warning("Stage-3.1 completed but mc_summary.json not found at %s", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"stage3_1_mc run_id: {summary['run_id']}")
    for method_key in methods:
        payload = summary["methods"].get(method_key)
        if payload is None:
            continue
        mc = payload["summary"]
        print(
            f"{method_key}: trade_count_source={int(payload['trade_count_source'])}, "
            f"return_p05={float(mc['return_pct']['p05']):.4f}, return_median={float(mc['return_pct']['median']):.4f}, "
            f"return_p95={float(mc['return_pct']['p95']):.4f}, maxDD_p95={float(mc['max_drawdown']['p95']):.4f}, "
            f"maxDD_p99={float(mc['max_drawdown']['p99']):.4f}, P(return<0)={float(mc['tail_probabilities']['p_return_lt_0']):.4f}, "
            f"P(ruin)={float(mc['tail_probabilities']['p_ruin']):.4f}"
        )
    print(f"recommendation: {summary['recommendation']}")
    print(f"mc_report: {run_dir / 'mc_report.md'}")


if __name__ == "__main__":
    main()
