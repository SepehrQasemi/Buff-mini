"""Run Stage-15..22 pipeline and build master report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.alpha_v2.master import build_master_summary, write_master_report


STAGE_SCRIPTS = [
    ("15", "scripts/run_stage15.py"),
    ("16", "scripts/run_stage16.py"),
    ("17", "scripts/run_stage17.py"),
    ("18", "scripts/run_stage18.py"),
    ("19", "scripts/run_stage19.py"),
    ("20", "scripts/run_stage20.py"),
    ("21", "scripts/run_stage21.py"),
    ("22", "scripts/run_stage22.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-15..22 pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-runs", action="store_true", help="Only rebuild master report from existing stage summaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_runs:
        for stage, script in STAGE_SCRIPTS:
            cmd = [sys.executable, script, "--seed", str(int(args.seed))]
            if bool(args.dry_run):
                cmd.append("--dry-run")
            subprocess.run(cmd, check=True)

    summaries = {stage: _load_json(Path(f"docs/stage{stage}_summary.json")) for stage, _ in STAGE_SCRIPTS}
    master_summary = build_master_summary(
        summaries=summaries,
        seed=int(args.seed),
        git_head=_git_head(),
        commit_hashes=_stage_commit_hashes(),
    )
    write_master_report(master_summary)
    print("master_summary: docs/stage15_22_master_summary.json")
    print("master_report: docs/stage15_22_master_report.md")
    print(f"final_verdict: {master_summary['final_verdict']}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return ""


def _stage_commit_hashes() -> list[str]:
    try:
        lines = subprocess.check_output(
            ["git", "log", "--oneline", "--max-count", "200"],
            text=True,
        ).splitlines()
    except Exception:
        return []
    prefixes = tuple(f"Stage-{x}" for x in ("15", "16", "17", "18", "19", "20", "21", "22"))
    hashes: list[str] = []
    for line in lines:
        if " " not in line:
            continue
        commit_hash, subject = line.split(" ", 1)
        if any(subject.startswith(prefix) for prefix in prefixes):
            hashes.append(commit_hash)
    return hashes


if __name__ == "__main__":
    main()
