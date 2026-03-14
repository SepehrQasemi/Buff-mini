"""Run Stage-60..72 sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-60..72 chain")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--campaign-runs", type=int, default=20)
    parser.add_argument(
        "--skip-stage51-59",
        action="store_true",
        default=False,
        help="Skip Stage-51..59 execution (assumes artifacts already exist).",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _load_stage28_run_id(docs_dir: Path) -> str:
    stage60_path = docs_dir / "stage60_summary.json"
    if not stage60_path.exists():
        return ""
    payload = json.loads(stage60_path.read_text(encoding="utf-8"))
    return str(payload.get("stage28_run_id", "")).strip()


def _resolve_stage28_from_docs(docs_dir: Path) -> str:
    candidates = [
        docs_dir / "stage39_signal_generation_summary.json",
        docs_dir / "stage47_signal_gen2_summary.json",
        docs_dir / "stage48_tradability_learning_summary.json",
        docs_dir / "stage52_summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = str(payload.get("stage28_run_id", "")).strip()
        if value:
            return value
    return ""


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable]
    stage28_run_id = _resolve_stage28_from_docs(docs_dir)
    if stage28_run_id and not bool(args.skip_stage51_59):
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
                "--allow-carryover",
                "--stage28-run-id",
                stage28_run_id,
            ]
        )
    _run([*base, "scripts/run_stage60.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir)])
    stage28_run_id = _load_stage28_run_id(docs_dir)
    if not stage28_run_id:
        raise SystemExit("stage60 did not resolve stage28_run_id")
    for stage in ("62", "66", "67", "68", "69"):
        _run(
            [
                *base,
                f"scripts/run_stage{stage}.py",
                "--config",
                str(args.config),
                "--runs-dir",
                str(args.runs_dir),
                "--docs-dir",
                str(docs_dir),
                "--stage28-run-id",
                stage28_run_id,
            ]
        )
    _run([*base, "scripts/run_stage63.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage64.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage65.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage70.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage71.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    # Re-materialize decision chain after real validation artifacts (stage67/68/69/71) exist.
    _run([*base, "scripts/run_stage61.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage57.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage58.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage59.py", "--config", str(args.config), "--docs-dir", str(docs_dir)])
    _run([*base, "scripts/run_stage72.py", "--config", str(args.config), "--docs-dir", str(docs_dir), "--campaign-runs", str(int(args.campaign_runs))])
    from buffmini.diagnostics import build_full_trace_report, write_full_trace_report

    _ = build_full_trace_report(docs_dir=docs_dir, runs_dir=Path(args.runs_dir), config_path=Path(args.config))
    write_full_trace_report(docs_dir=docs_dir, runs_dir=Path(args.runs_dir), config_path=Path(args.config))


if __name__ == "__main__":
    main()
