"""Run Stage-82 failure-driven search learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.research.learning import build_edge_inventory, build_failure_taxonomy, derive_search_feedback
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-82 failure-driven search learning")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    ranking_cards = _read_csv(docs_dir / "stage79_ranking_cards.csv")
    stage67 = _load_json(docs_dir / "stage67_summary.json")
    stage81 = _load_json(docs_dir / "stage81_summary.json")
    taxonomy = build_failure_taxonomy(ranking_cards=ranking_cards, stage67_summary=stage67, stage81_summary=stage81)
    inventory = build_edge_inventory(ranking_cards)
    feedback = derive_search_feedback(ranking_cards=ranking_cards, failure_taxonomy=taxonomy)
    feedback_path = docs_dir / "stage82_search_feedback.json"
    feedback_path.write_text(json.dumps(feedback, indent=2, allow_nan=False), encoding="utf-8")

    status = "SUCCESS" if inventory or any(int(value) > 0 for value in taxonomy.values()) else "PARTIAL"
    summary = {
        "stage": "82",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "orchestration_only",
        "validation_state": "FAILURE_FEEDBACK_READY" if status == "SUCCESS" else "FAILURE_FEEDBACK_EMPTY",
        "failure_taxonomy": taxonomy,
        "edge_inventory": inventory,
        "search_feedback": feedback,
        "feedback_artifact_path": str(feedback_path).replace("\\", "/"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage82_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-82 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        "",
        "## Failure Taxonomy",
    ]
    for key, value in sorted(taxonomy.items()):
        lines.append(f"- {key}: `{int(value)}`")
    lines.extend(["", "## Search Feedback"])
    for key, value in sorted((feedback.get("family_priority_adjustments") or {}).items()):
        lines.append(f"- {key}: `{float(value):.6f}`")
    lines.extend(["", f"- threshold_guidance: `{feedback.get('threshold_guidance', [])}`", "", f"- summary_hash: `{summary['summary_hash']}`"])
    (docs_dir / "stage82_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
