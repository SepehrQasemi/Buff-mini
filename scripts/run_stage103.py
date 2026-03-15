"""Run Stage-103 final edge existence verdict aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts
from buffmini.research.verdict import derive_final_edge_verdict, load_stage_payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-103 final edge existence verdict")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = load_stage_payloads(Path(args.docs_dir), stages=list(range(95, 103)))
    summary = derive_final_edge_verdict(payloads)
    summary.update(
        {
            "stage": "103",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "reporting_only",
            "validation_state": "FINAL_EDGE_VERDICT_READY",
        }
    )
    lines = [
        "# Stage-103 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- final_edge_verdict: `{summary['final_edge_verdict']}`",
        "",
    ]
    lines.extend(markdown_kv("Supporting Counts", dict(summary.get("supporting_counts", {}))))
    lines.extend([""] + markdown_rows("Evidence Table", list(summary.get("evidence_table", [])), limit=24))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="103", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
