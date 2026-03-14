"""Run Stage-63 free data mesh planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage63 import build_free_data_mesh_plan
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-63 free data mesh")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    plan = build_free_data_mesh_plan(cfg)
    status = "SUCCESS" if len(plan["source_contracts"]) >= 3 else "PARTIAL"
    summary = {
        "stage": "63",
        "status": status,
        "enabled_sources": plan["enabled_sources"],
        "source_priority": plan["source_priority"],
        "source_count": len(plan["source_contracts"]),
        "families_covered": sorted(plan["schema"].keys()),
        "blocker_reason": "" if status == "SUCCESS" else "insufficient_free_sources",
    }
    summary["summary_hash"] = stable_hash(
        {"summary": summary, "plan_hash": plan["summary_hash"]},
        length=16,
    )
    (docs_dir / "stage63_data_mesh_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage63_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage63_report.md").write_text(
        "\n".join(
            [
                "# Stage-63 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- enabled_sources: `{summary['enabled_sources']}`",
                f"- source_priority: `{summary['source_priority']}`",
                f"- source_count: `{summary['source_count']}`",
                f"- families_covered: `{summary['families_covered']}`",
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

