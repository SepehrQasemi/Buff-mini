"""Run Stage-51 to Stage-59 sequentially, with optional full fresh upstream chain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-51..59 chain")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-carryover",
        action="store_true",
        default=False,
        help="Bypass fresh-start policy and reuse existing chain artifacts (not recommended).",
    )
    parser.add_argument("--stage28-mode", type=str, choices=["research", "live", "both"], default="both")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(cli_value: str, docs_dir: Path) -> str:
    if str(cli_value).strip():
        return str(cli_value).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    if str(stage39.get("stage28_run_id", "")).strip():
        return str(stage39.get("stage28_run_id", "")).strip()
    stage28 = _load_json(docs_dir / "stage28_master_summary.json")
    return str(stage28.get("run_id", "")).strip()


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return str(result.stdout or "")


def _extract_line_value(text: str, prefix: str) -> str:
    for raw in str(text).splitlines():
        line = str(raw).strip()
        if line.lower().startswith(prefix.lower()):
            return str(line[len(prefix) :]).strip()
    return ""


def _cleanup_fresh_docs(docs_dir: Path) -> None:
    for name in (
        "stage39_signal_generation_summary.json",
        "stage39_signal_generation_report.md",
        "stage47_signal_gen2_summary.json",
        "stage47_signal_gen2_report.md",
        "stage48_tradability_learning_summary.json",
        "stage48_tradability_learning_report.md",
        "stage51_summary.json",
        "stage51_report.md",
        "stage52_summary.json",
        "stage52_report.md",
        "stage53_summary.json",
        "stage53_report.md",
        "stage54_summary.json",
        "stage54_report.md",
        "stage55_summary.json",
        "stage55_report.md",
        "stage56_summary.json",
        "stage56_report.md",
        "stage57_summary.json",
        "stage57_report.md",
        "stage57_history.json",
        "stage57_chain_metrics.json",
        "stage58_summary.json",
        "stage58_report.md",
        "stage59_summary.json",
        "stage59_report.md",
    ):
        path = docs_dir / name
        if path.exists():
            path.unlink()


def _run_upstream_chain(*, args: argparse.Namespace, docs_dir: Path) -> str:
    base = [sys.executable]
    stage28_cmd = [
        *base,
        "scripts/run_stage28.py",
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seed)),
        "--mode",
        str(args.stage28_mode),
        "--runs-dir",
        str(args.runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    out = _run(stage28_cmd)
    stage28_run_id = _extract_line_value(out, "run_id:")
    if not stage28_run_id:
        stage28_summary = _load_json(docs_dir / "stage28_master_summary.json")
        stage28_run_id = str(stage28_summary.get("run_id", "")).strip()
    if not stage28_run_id:
        raise SystemExit("failed to resolve stage28_run_id from Stage-28 output")

    _run(
        [
            *base,
            "scripts/run_stage39.py",
            "--seed",
            str(int(args.seed)),
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
            "--stage28-run-id",
            stage28_run_id,
        ]
    )
    _run(
        [
            *base,
            "scripts/run_stage47.py",
            "--seed",
            str(int(args.seed)),
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
            "--stage28-run-id",
            stage28_run_id,
        ]
    )
    _run(
        [
            *base,
            "scripts/run_stage48.py",
            "--config",
            str(args.config),
            "--seed",
            str(int(args.seed)),
            "--runs-dir",
            str(args.runs_dir),
            "--docs-dir",
            str(docs_dir),
            "--stage28-run-id",
            stage28_run_id,
        ]
    )
    return stage28_run_id


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage28_run_id = _resolve_stage28_run_id(args.stage28_run_id, docs_dir)
    if bool(args.allow_carryover):
        if not stage28_run_id:
            raise SystemExit("allow-carryover requires --stage28-run-id or resolvable docs summaries")
    else:
        if not bool(args.fresh_start):
            raise SystemExit("carryover mode is disabled by policy; remove --no-fresh-start")
        _cleanup_fresh_docs(docs_dir)
        stage28_run_id = _run_upstream_chain(args=args, docs_dir=docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id")

    base = [sys.executable]
    chain: list[list[str]] = [
        [*base, "scripts/run_stage51.py", "--config", str(args.config), "--docs-dir", str(docs_dir)],
        [*base, "scripts/run_stage52.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir), "--stage28-run-id", stage28_run_id],
        [*base, "scripts/run_stage53.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir), "--stage28-run-id", stage28_run_id],
        [*base, "scripts/run_stage54.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir), "--stage28-run-id", stage28_run_id],
        [*base, "scripts/run_stage55.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir), "--stage28-run-id", stage28_run_id],
        [*base, "scripts/run_stage56.py", "--config", str(args.config), "--runs-dir", str(args.runs_dir), "--docs-dir", str(docs_dir), "--stage28-run-id", stage28_run_id],
        [*base, "scripts/run_stage57.py", "--config", str(args.config), "--docs-dir", str(docs_dir)],
        [*base, "scripts/run_stage58.py", "--config", str(args.config), "--docs-dir", str(docs_dir)],
        [*base, "scripts/run_stage59.py", "--config", str(args.config), "--docs-dir", str(docs_dir)],
    ]

    for cmd in chain:
        _run(cmd)


if __name__ == "__main__":
    main()
