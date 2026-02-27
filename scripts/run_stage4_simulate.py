"""Run Stage-4 offline paper-execution simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.execution.simulator import run_stage4_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-4 execution simulator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--stage3-3-run-id", type=str, default=None)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage3_summary = None
    if args.stage3_3_run_id:
        stage3_summary = _load_json(args.runs_dir / args.stage3_3_run_id / "selector_summary.json")

    run_dir = run_stage4_simulation(
        stage2_run_id=args.stage2_run_id,
        cfg=config,
        stage3_choice=stage3_summary,
        days=args.days,
        bars=args.bars,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    metrics = _load_json(run_dir / "execution_metrics.json")
    print(f"stage4_sim_run: {run_dir.name}")
    print(f"method: {metrics['method']}")
    print(f"leverage: {metrics['chosen_leverage']}")
    print(f"order_count: {metrics['metrics']['order_count']}")
    print(f"scaled_event_count: {metrics['metrics']['scaled_event_count']}")
    print(f"killswitch_event_count: {metrics['metrics']['killswitch_event_count']}")
    print(f"artifacts: {run_dir}")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

