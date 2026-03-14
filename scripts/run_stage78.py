"""Run Stage-78 mechanism-based generator redesign audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage51 import resolve_budget_mode, resolve_research_scope
from buffmini.stage70 import generate_expanded_candidates
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-78 mechanism-based generator audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    scope = resolve_research_scope(cfg)
    budget = resolve_budget_mode(cfg)
    target = 10000 if str(budget["selected"]).strip().lower() == "full_audit" else 2500
    frame = generate_expanded_candidates(
        discovery_timeframes=[str(v) for v in scope["discovery_timeframes"]],
        budget_mode_selected=str(budget["selected"]),
        active_families=list(scope["active_setup_families"]),
    )
    frame.to_csv(docs_dir / "stage78_mechanism_candidates.csv", index=False)
    family_counts = {str(key): int(value) for key, value in frame["family"].astype(str).value_counts(dropna=False).to_dict().items()} if not frame.empty else {}
    mechanism_diversity = float(round(frame["mechanism_signature"].astype(str).nunique() / max(1, len(frame)), 6)) if not frame.empty else 0.0
    summary = {
        "stage": "78",
        "status": "SUCCESS" if int(len(frame)) >= int(target) and int(frame["family"].nunique()) >= 3 and mechanism_diversity >= 0.95 else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "heuristic_filter",
        "validation_state": "MECHANISM_GENERATION_READY" if int(len(frame)) >= int(target) and int(frame["family"].nunique()) >= 3 else "MECHANISM_GENERATION_WEAK",
        "target_candidate_count": int(target),
        "candidate_count": int(len(frame)),
        "mechanism_family_coverage": family_counts,
        "family_specific_candidate_counts": family_counts,
        "mechanism_diversity": mechanism_diversity,
        "risk_model_count": int(frame["risk_model"].astype(str).nunique()) if not frame.empty else 0,
        "exit_family_count": int(frame["exit_family"].astype(str).nunique()) if not frame.empty else 0,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage78_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-78 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- target_candidate_count: `{summary['target_candidate_count']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- mechanism_diversity: `{summary['mechanism_diversity']}`",
        f"- risk_model_count: `{summary['risk_model_count']}`",
        f"- exit_family_count: `{summary['exit_family_count']}`",
        "",
        "## Mechanism Family Coverage",
    ]
    for family, count in sorted(family_counts.items()):
        lines.append(f"- {family}: `{count}`")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    (docs_dir / "stage78_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
