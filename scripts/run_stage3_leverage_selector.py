"""Run Stage-3.3 automatic leverage selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.leverage_selector import run_stage3_leverage_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-3.3 automatic leverage selector")
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--leverage-levels", type=str, default=None)
    parser.add_argument("--n-paths", type=int, default=None)
    parser.add_argument("--bootstrap", type=str, choices=["iid", "block"], default=None)
    parser.add_argument("--block-size-trades", type=int, default=None)
    parser.add_argument("--ruin-dd-threshold", type=float, default=None)
    parser.add_argument("--max-p-ruin", type=float, default=None)
    parser.add_argument("--max-dd-p95", type=float, default=None)
    parser.add_argument("--min-return-p05", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selector_cfg: dict[str, Any] = dict(config["portfolio"]["leverage_selector"])

    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()] if args.methods else None
    leverage_levels = (
        [float(item.strip()) for item in str(args.leverage_levels).split(",") if item.strip()]
        if args.leverage_levels
        else None
    )

    command = (
        "python scripts/run_stage3_leverage_selector.py "
        f"--stage2-run-id {args.stage2_run_id}"
    )
    for key, value in [
        ("--methods", args.methods),
        ("--leverage-levels", args.leverage_levels),
        ("--n-paths", args.n_paths),
        ("--bootstrap", args.bootstrap),
        ("--block-size-trades", args.block_size_trades),
        ("--ruin-dd-threshold", args.ruin_dd_threshold),
        ("--max-p-ruin", args.max_p_ruin),
        ("--max-dd-p95", args.max_dd_p95),
        ("--min-return-p05", args.min_return_p05),
        ("--seed", args.seed),
        ("--initial-equity", args.initial_equity),
    ]:
        if value is not None:
            command += f" {key} {value}"

    run_dir = run_stage3_leverage_selector(
        stage2_run_id=args.stage2_run_id,
        selector_cfg=selector_cfg,
        methods=methods,
        leverage_levels=leverage_levels,
        seed=args.seed,
        n_paths=args.n_paths,
        bootstrap=args.bootstrap,
        block_size_trades=args.block_size_trades,
        initial_equity=args.initial_equity,
        ruin_dd_threshold=args.ruin_dd_threshold,
        max_p_ruin=args.max_p_ruin,
        max_dd_p95=args.max_dd_p95,
        min_return_p05=args.min_return_p05,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        cli_command=command,
    )

    summary = json.loads((run_dir / "selector_summary.json").read_text(encoding="utf-8"))
    print(f"stage3_3 run_id: {summary['run_id']}")
    for method, payload in summary["method_choices"].items():
        chosen = payload.get("chosen_row")
        if chosen is None:
            print(
                f"{method}: chosen_L=None, expected_log_growth=None, "
                f"binding_constraints={payload.get('first_failure_constraints', [])}"
            )
        else:
            print(
                f"{method}: chosen_L={payload['chosen_leverage']}, "
                f"expected_log_growth={float(chosen['expected_log_growth']):.6f}, "
                f"binding_constraints={payload.get('binding_constraints', [])}"
            )
    overall = summary["overall_choice"]
    print(f"overall: method={overall['method']}, leverage={overall['chosen_leverage']}")
    print(f"selector_report: {run_dir / 'selector_report.md'}")


if __name__ == "__main__":
    main()

