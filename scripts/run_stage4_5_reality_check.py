"""Run Stage-4.5 Reality Check robustness layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from buffmini.constants import RUNS_DIR
from buffmini.execution.reality_check import RealityCheckConfig, run_reality_check
from buffmini.ui.components.run_index import latest_completed_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-4.5 reality check")
    parser.add_argument("--run-id", type=str, default=None, help="Pipeline run id (defaults to latest completed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_id = args.run_id
    if not run_id:
        latest = latest_completed_pipeline(args.runs_dir)
        if latest is None:
            raise FileNotFoundError("No completed pipeline runs found")
        run_id = str(latest["run_id"])

    rc_dir = run_reality_check(
        run_id=run_id,
        runs_dir=args.runs_dir,
        cfg=RealityCheckConfig(seed=int(args.seed)),
    )

    summary_path = rc_dir / "reality_check_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    print(f"run_id: {run_id}")
    print(f"reality_check_dir: {rc_dir}")
    print(f"confidence_score: {float(payload['confidence_score']):.6f}")
    print(f"verdict: {payload['verdict']}")
    print(f"summary_sha256: {sha}")


if __name__ == "__main__":
    main()
