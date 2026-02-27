"""Run Stage-2.5 rolling walk-forward portfolio validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.constants import (
    DEFAULT_WALKFORWARD_FORWARD_DAYS,
    DEFAULT_WALKFORWARD_NUM_WINDOWS,
    RAW_DATA_DIR,
    RUNS_DIR,
)
from buffmini.portfolio.walkforward import run_stage2_walkforward
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-2.5 walk-forward portfolio validation")
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--forward-days", type=int, default=DEFAULT_WALKFORWARD_FORWARD_DAYS)
    parser.add_argument("--num-windows", type=int, default=DEFAULT_WALKFORWARD_NUM_WINDOWS)
    parser.add_argument("--reserve-forward-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = (
        "python scripts/run_stage2_walkforward.py "
        f"--stage2-run-id {args.stage2_run_id} --forward-days {int(args.forward_days)} "
        f"--num-windows {int(args.num_windows)} --seed {int(args.seed)}"
    )
    if args.reserve_forward_days is not None:
        command += f" --reserve-forward-days {int(args.reserve_forward_days)}"
    run_dir = run_stage2_walkforward(
        stage2_run_id=args.stage2_run_id,
        forward_days=args.forward_days,
        num_windows=args.num_windows,
        seed=args.seed,
        reserve_forward_days=args.reserve_forward_days,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        cli_command=command,
    )

    summary_path = run_dir / "walkforward_summary.json"
    if not summary_path.exists():
        logger.warning("Stage-2.5 completed but walkforward_summary.json not found at %s", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Stage-2.5 run_id: {summary['run_id']}")
    print(f"recommendation: {summary['overall_recommendation']}")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        stability = payload["stability"]
        forward_pf_list = [
            float(item["metrics"]["profit_factor"])
            for item in payload["window_metrics"]
            if str(item["window"]).startswith("Forward")
        ]
        print(
            f"{method_key}: pf_holdout={float(stability['pf_holdout']):.4f}, "
            f"pf_forward_mean={float(stability['pf_forward_mean']):.4f}, "
            f"degradation_ratio={float(stability['degradation_ratio']):.4f}, "
            f"worst_forward_pf={float(stability['worst_forward_pf']):.4f}, "
            f"dd_growth_ratio={float(stability['dd_growth_ratio']):.4f}, "
            f"usable_windows={int(stability['usable_windows'])}, "
            f"classification={stability['classification']}, "
            f"forward_pfs={forward_pf_list}"
        )
    print(f"walkforward_report: {run_dir / 'walkforward_report.md'}")


if __name__ == "__main__":
    main()
