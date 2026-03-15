"""Run Stage-91 controlled scope expansion ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.research.scope_ladder import evaluate_scope_ladder
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-91 controlled scope expansion ladder")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--candidate-limit-per-scope", type=int, default=3)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    feedback = _load_json(Path(args.docs_dir) / "stage82_search_feedback.json")
    ladder = evaluate_scope_ladder(cfg, feedback=feedback, candidate_limit=int(args.candidate_limit_per_scope))
    summary = {
        "stage": "91",
        "status": "SUCCESS",
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "SCOPE_LADDER_READY",
        **ladder,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-91 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- tier1_symbols: `{summary['tier1_symbols']}`",
        f"- tier2_symbols: `{summary['tier2_symbols']}`",
        f"- candidate_limit_per_scope: `{summary['candidate_limit_per_scope']}`",
        f"- row_count: `{len(summary['rows'])}`",
    ]
    lines.extend([""] + markdown_rows("Scope Ladder Rows", list(summary.get("rows", [])), limit=16))
    lines.extend([""] + markdown_rows("Regime Scope Map", list(summary.get("regime_scope_map", [])), limit=8))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="91", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
