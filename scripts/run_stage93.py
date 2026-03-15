"""Run Stage-93 failure-driven learning loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.funnel import analyze_funnel_pressure
from buffmini.research.learning import build_failure_taxonomy, build_traceable_learning_loop
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-93 failure-driven learning loop")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    cfg = load_config(Path(args.config))
    stage88 = _load_json(docs_dir / "stage88_summary.json")
    if not stage88:
        stage88 = analyze_funnel_pressure(cfg)
    stage67 = _load_json(docs_dir / "stage67_summary.json")
    stage92 = _load_json(docs_dir / "stage92_summary.json")
    ranking_cards = pd.DataFrame(list(stage88.get("evaluations", [])))
    stage81_like = {
        "transfer_matrix_rows": int(len(stage92.get("transfer_rows", []))),
        "transfer_class_counts": dict(stage92.get("transfer_class_counts", {})),
    }
    failure_taxonomy = build_failure_taxonomy(
        ranking_cards=ranking_cards,
        stage67_summary=stage67,
        stage81_summary=stage81_like,
    )
    success_inventory = [
        row
        for row in ranking_cards.to_dict(orient="records")
        if str(row.get("candidate_hierarchy", "")) in {"promising_but_unproven", "validated_candidate", "robust_candidate"}
    ]
    loop = build_traceable_learning_loop(
        failure_taxonomy=failure_taxonomy,
        ranking_cards=ranking_cards,
        stage92_summary=stage92,
        success_inventory=success_inventory,
    )
    summary = {
        "stage": "93",
        "status": "SUCCESS",
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "FAILURE_LEARNING_READY",
        **loop,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage93_search_feedback.json").write_text(json.dumps(dict(summary.get("search_feedback", {})), indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-93 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- success_inventory_count: `{summary['success_inventory_count']}`",
        f"- learning_trace_hash: `{summary['learning_trace_hash']}`",
    ]
    lines.extend([""] + markdown_rows("Failure Taxonomy", [{"failure": key, "count": value} for key, value in dict(summary.get("failure_taxonomy", {})).items()], limit=12))
    lines.extend([""] + markdown_rows("Adaptation Steps", list(summary.get("adaptation_steps", [])), limit=12))
    lines.extend([""] + markdown_rows("Search Feedback", [dict(summary.get("search_feedback", {}))], limit=1))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=docs_dir, stage="93", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
