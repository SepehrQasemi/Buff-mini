"""Run Stage-83 research operations maturity summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.research.ops import build_human_review_checklist, build_run_registry_entry
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-83 research operations maturity summary")
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
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage77 = _load_json(docs_dir / "stage77_summary.json")
    stage67 = _load_json(docs_dir / "stage67_summary.json")
    stage80 = _load_json(docs_dir / "stage80_summary.json")
    stage81 = _load_json(docs_dir / "stage81_summary.json")
    stage82 = _load_json(docs_dir / "stage82_summary.json")
    run_id = str(stage67.get("stage28_run_id", ""))

    registry_entry = build_run_registry_entry(
        run_id=run_id,
        mode_summary=stage77,
        stage80_summary=stage80,
        stage81_summary=stage81,
        stage82_summary=stage82,
    )
    checklist = build_human_review_checklist(
        mode_summary=stage77,
        stage67_summary=stage67,
        stage80_summary=stage80,
        stage81_summary=stage81,
    )
    registry_path = docs_dir / "run_registry.json"
    existing = []
    if registry_path.exists():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        existing = payload if isinstance(payload, list) else []
    deduped = [row for row in existing if str(row.get("run_id", "")) != str(run_id)]
    deduped.append(registry_entry)
    registry_path.write_text(json.dumps(deduped, indent=2, allow_nan=False), encoding="utf-8")

    summary = {
        "stage": "83",
        "status": "SUCCESS",
        "execution_status": "EXECUTED",
        "stage_role": "orchestration_only",
        "validation_state": "RESEARCH_OPS_READY",
        "run_registry_entry": registry_entry,
        "checklist": checklist,
        "registry_path": str(registry_path).replace("\\", "/"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage83_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-83 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- registry_path: `{summary['registry_path']}`",
        "",
        "## Run Registry Entry",
    ]
    for key, value in summary["run_registry_entry"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Human Review Checklist"])
    for row in checklist:
        lines.append(f"- {row['item']}: `{bool(row['passed'])}` ({row['reason']})")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    (docs_dir / "stage83_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
