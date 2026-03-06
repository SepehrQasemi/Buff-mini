"""Stage-38.1 runtime execution tracing for Stage-28/Stage-37 lineage."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage38.reporting import render_stage38_flow_report
from buffmini.stage38.trace import build_stage28_execution_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-38 end-to-end execution trace report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--mode", type=str, default="both", choices=["research", "live", "both"])
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _run_stage28(*, config: Path, seed: int, mode: str, budget_small: bool, runs_dir: Path, docs_dir: Path) -> str:
    cmd = [
        sys.executable,
        "scripts/run_stage28.py",
        "--config",
        str(config),
        "--seed",
        str(int(seed)),
        "--mode",
        str(mode),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if bool(budget_small):
        cmd.append("--budget-small")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if int(proc.returncode) != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-80:])
        raise RuntimeError(f"run_stage28 failed exit={proc.returncode}\n{tail}")
    match = re.search(r"^run_id:\s*(\S+)\s*$", stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("run_stage28 output missing run_id")
    return str(match.group(1))


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage28_run_id = str(args.stage28_run_id).strip()
    if not stage28_run_id:
        stage28_run_id = _run_stage28(
            config=Path(args.config),
            seed=int(args.seed),
            mode=str(args.mode),
            budget_small=bool(args.budget_small),
            runs_dir=Path(args.runs_dir),
            docs_dir=docs_dir,
        )

    stage28_dir = Path(args.runs_dir) / stage28_run_id / "stage28"
    if not stage28_dir.exists():
        raise SystemExit(f"missing stage28 directory: {stage28_dir}")

    payload = build_stage28_execution_trace(stage28_dir=stage28_dir, config_path=Path(args.config))
    out_dir = Path(args.runs_dir) / stage28_run_id / "stage38"
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_json = out_dir / "stage38_trace.json"
    trace_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    report_md = docs_dir / "stage38_end_to_end_flow_report.md"
    report_md.write_text(render_stage38_flow_report(payload), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"trace_json: {trace_json}")
    print(f"flow_report: {report_md}")
    print(f"trace_hash: {payload.get('trace_hash', '')}")


if __name__ == "__main__":
    main()

