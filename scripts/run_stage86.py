"""Run Stage-86 family coverage audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.family_audit import evaluate_family_audit
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-86 family coverage audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    audit = evaluate_family_audit(cfg)
    summary = {
        "stage": "86",
        "status": "SUCCESS" if audit.get("family_count", 0) > 0 else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "FAMILY_AUDIT_READY" if audit.get("family_count", 0) > 0 else "FAMILY_AUDIT_INCOMPLETE",
        **audit,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-86 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- family_count: `{summary['family_count']}`",
        f"- subfamily_total: `{summary['subfamily_total']}`",
        f"- validation_state: `{summary['validation_state']}`",
    ]
    lines.extend([""] + markdown_rows("Family Inventory", list(summary.get("family_inventory", [])), limit=8))
    lines.extend([""] + markdown_rows("Blind Spots", list(summary.get("blind_spots", [])), limit=12))
    overlap = dict(summary.get("overlap_analysis", {}))
    lines.extend([""] + markdown_rows("Top Semantic Overlap", list(overlap.get("semantic_overlap_pairs", [])), limit=8))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="86", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
