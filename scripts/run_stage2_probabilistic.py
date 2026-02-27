"""Run Stage-2.8 probabilistic rolling walk-forward evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.probabilistic import run_stage2_probabilistic
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-2.8 probabilistic rolling walk-forward evaluation")
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--stride-days", type=int, default=7)
    parser.add_argument("--num-windows", type=int, default=None)
    parser.add_argument("--reserve-tail-days", type=int, default=180)
    parser.add_argument("--min_trades", type=int, default=20)
    parser.add_argument("--min_exposure", type=float, default=0.01)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = (
        "python scripts/run_stage2_probabilistic.py "
        f"--stage2-run-id {args.stage2_run_id} --window-days {int(args.window_days)} "
        f"--stride-days {int(args.stride_days)} --reserve-tail-days {int(args.reserve_tail_days)} "
        f"--min_trades {int(args.min_trades)} --min_exposure {float(args.min_exposure)} "
        f"--n_boot {int(args.n_boot)} --seed {int(args.seed)}"
    )
    if args.num_windows is not None:
        command += f" --num-windows {int(args.num_windows)}"

    run_dir = run_stage2_probabilistic(
        stage2_run_id=args.stage2_run_id,
        window_days=args.window_days,
        stride_days=args.stride_days,
        num_windows=args.num_windows,
        reserve_tail_days=args.reserve_tail_days,
        min_trades=args.min_trades,
        min_exposure=args.min_exposure,
        n_boot=args.n_boot,
        seed=args.seed,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        cli_command=command,
    )

    summary_path = run_dir / "probabilistic_summary.json"
    if not summary_path.exists():
        logger.warning("Stage-2.8 completed but probabilistic_summary.json not found at %s", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Stage-2.8 run_id: {summary['run_id']}")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        aggregate = payload["aggregate"]
        print(
            f"{method_key}: usable_windows={int(aggregate['usable_windows'])}/{int(aggregate['total_windows'])}, "
            f"p_edge_gt0_median={float(aggregate['p_edge_gt0_median']):.4f}, "
            f"p_pf_gt1_median={float(aggregate['p_pf_gt1_median']):.4f}, "
            f"robustness_score={float(aggregate['robustness_score']):.4f}, "
            f"classification={aggregate['classification']}"
        )
    print(f"final_recommendation: {summary['overall_recommendation']}")
    print(f"probabilistic_report: {run_dir / 'probabilistic_report.md'}")


if __name__ == "__main__":
    main()
