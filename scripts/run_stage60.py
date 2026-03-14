"""Run Stage-60 chain integrity lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage60 import assess_chain_integrity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-60 chain integrity lock")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _render(summary: dict[str, object]) -> str:
    lines = [
        "# Stage-60 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary.get('execution_status', '')}`",
        f"- stage_role: `{summary.get('stage_role', '')}`",
        f"- stage28_run_id: `{summary['stage28_run_id']}`",
        f"- chain_id: `{summary['chain_id']}`",
        f"- budget_mode_selected: `{summary['budget_mode_selected']}`",
        f"- missing_summaries: `{summary['missing_summaries']}`",
        f"- run_id_mismatch: `{summary['run_id_mismatch']}`",
        f"- missing_artifacts: `{summary['missing_artifacts']}`",
        f"- bootstrap_forbidden: `{summary['bootstrap_forbidden']}`",
        f"- bootstrap_stages: `{summary['bootstrap_stages']}`",
        f"- blocker_reason: `{summary['blocker_reason']}`",
        f"- summary_hash: `{summary['summary_hash']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    budget_mode = str(cfg.get("budget_mode", {}).get("selected", "search"))
    summary = assess_chain_integrity(
        docs_dir=docs_dir,
        runs_dir=Path(args.runs_dir),
        budget_mode_selected=budget_mode,
    )
    summary["execution_status"] = "EXECUTED"
    summary["stage_role"] = "orchestration_only"
    summary_path = docs_dir / "stage60_summary.json"
    report_path = docs_dir / "stage60_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(summary), encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
