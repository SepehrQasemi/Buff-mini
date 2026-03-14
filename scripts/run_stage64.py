"""Run Stage-64 backfill + coverage planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage51 import resolve_research_scope
from buffmini.stage64 import build_backfill_plan_v2
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-64 backfill planner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage63_path = docs_dir / "stage63_data_mesh_plan.json"
    stage63_plan = json.loads(stage63_path.read_text(encoding="utf-8")) if stage63_path.exists() else {}
    scope = resolve_research_scope(cfg)
    plan = build_backfill_plan_v2(
        source_contracts=[dict(v) for v in stage63_plan.get("source_contracts", []) if isinstance(v, dict)],
        active_families=[str(v) for v in scope.get("active_setup_families", [])],
    )
    matrix = plan["source_coverage_matrix"]
    matrix_path = docs_dir / "stage64_source_coverage_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    staleness_path = docs_dir / "stage64_staleness_report.json"
    staleness_path.write_text(json.dumps(plan["staleness_report"], indent=2, allow_nan=False), encoding="utf-8")

    summary = {
        "stage": "64",
        "status": plan["status"],
        "source_rows": int(len(matrix)),
        "disabled_families": plan["disabled_families"],
        "stale_sources": plan["staleness_report"]["stale_sources"],
        "fresh_sources": plan["staleness_report"]["fresh_sources"],
        "source_cost": 0.0,
        "blocker_reason": plan["blocker_reason"],
    }
    summary["summary_hash"] = stable_hash(
        {"summary": summary, "plan_hash": plan["summary_hash"]},
        length=16,
    )
    (docs_dir / "stage64_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage64_report.md").write_text(
        "\n".join(
            [
                "# Stage-64 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- source_rows: `{summary['source_rows']}`",
                f"- disabled_families: `{summary['disabled_families']}`",
                f"- stale_sources: `{summary['stale_sources']}`",
                f"- fresh_sources: `{summary['fresh_sources']}`",
                f"- source_cost: `{summary['source_cost']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()

