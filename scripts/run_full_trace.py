"""Run a full cold-start chain and emit a full trace report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full cold-start chain + trace report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage28-mode", type=str, choices=["research", "live", "both"], default="both")
    parser.add_argument("--campaign-runs", type=int, default=20)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable]

    _run(
        [
            *base,
            "scripts/run_stage51_59.py",
            "--config",
            str(args.config),
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
            "--seed",
            str(int(args.seed)),
            "--stage28-mode",
            str(args.stage28_mode),
        ]
    )

    _run(
        [
            *base,
            "scripts/run_stage60_72.py",
            "--config",
            str(args.config),
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
            "--campaign-runs",
            str(int(args.campaign_runs)),
            "--skip-stage51-59",
        ]
    )

    trace = _load_json(docs_dir / "full_trace_summary.json")
    summary_hash = str(trace.get("summary_hash", ""))
    print(f"trace_summary_hash: {summary_hash}")
    print(f"trace_report: {docs_dir / 'full_trace_report.md'}")


if __name__ == "__main__":
    main()

