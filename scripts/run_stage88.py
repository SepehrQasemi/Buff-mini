"""Run Stage-88 ranking and funnel pressure repair diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.funnel import analyze_funnel_pressure
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-88 ranking and funnel pressure diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    analysis = analyze_funnel_pressure(cfg)
    summary = {
        "stage": "88",
        "status": "SUCCESS" if analysis.get("candidate_count", 0) > 0 else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "heuristic_filter",
        "validation_state": "FUNNEL_PRESSURE_DIAGNOSED" if analysis.get("candidate_count", 0) > 0 else "FUNNEL_PRESSURE_INCOMPLETE",
        **analysis,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-88 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- symbol: `{summary['symbol']}`",
        f"- timeframe: `{summary['timeframe']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- promising_count: `{summary['promising_count']}`",
        f"- validated_count: `{summary['validated_count']}`",
        f"- robust_count: `{summary['robust_count']}`",
        f"- blocked_count: `{summary['blocked_count']}`",
        f"- dominant_culprit: `{summary['diagnosis']['dominant_culprit']}`",
    ]
    lines.extend([""] + markdown_rows("Candidate Hierarchy Counts", [{"class": key, "count": value} for key, value in dict(summary.get("candidate_hierarchy_counts", {})).items()]))
    lines.extend([""] + markdown_rows("Near Miss Inventory", list(summary.get("near_miss_inventory", [])), limit=12))
    lines.extend([""] + markdown_rows("Gate Heatmap", list(summary.get("gate_heatmap", [])), limit=18))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="88", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
