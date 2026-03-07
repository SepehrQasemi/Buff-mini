"""Stage-44 optimization framework core runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.stage44.contracts import (
    build_allocator_hook,
    build_contribution_record,
    build_failure_record,
    build_runtime_event,
    to_registry_row,
    validate_stage44_summary,
)
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-44 optimization framework core")
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _render(payload: dict[str, Any], *, notes: list[str]) -> str:
    lines = [
        "# Stage-44 Optimization Framework Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- contribution_contract_defined: `{bool(payload.get('contribution_contract_defined', False))}`",
        f"- failure_contract_defined: `{bool(payload.get('failure_contract_defined', False))}`",
        f"- runtime_contract_defined: `{bool(payload.get('runtime_contract_defined', False))}`",
        f"- allocator_hooks_defined: `{bool(payload.get('allocator_hooks_defined', False))}`",
        f"- registry_compatibility_defined: `{bool(payload.get('registry_compatibility_defined', False))}`",
        "",
        "## Modules Covered",
    ]
    for item in payload.get("modules_covered", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Remaining Gaps"])
    for item in payload.get("remaining_gaps", []):
        lines.append(f"- {item}")
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend([f"- {note}" for note in notes])
    lines.append("")
    lines.append(f"- summary_hash: `{payload.get('summary_hash', '')}`")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage41 = _load_json(docs_dir / "stage41_derivatives_completion_summary.json")
    stage42 = _load_json(docs_dir / "stage42_self_learning_2_summary.json")

    contrib_rows = list(stage41.get("family_contributions", []))
    if contrib_rows:
        first = dict(contrib_rows[0])
        family_name = str(first.get("family", "flow"))
        contribution = build_contribution_record(
            module_name="stage41_family_contribution_adapter",
            family_name=family_name,
            setup_name=f"{family_name}_base_setup",
            raw_candidate_contribution=float(first.get("candidate_lift", 0.0)),
            stage_a_survival_lift=float(first.get("activation_lift", 0.0)),
            stage_b_survival_lift=float(first.get("tradability_lift", 0.0)),
            final_policy_contribution=float(first.get("final_policy_share", 0.0)),
            runtime_seconds=0.001,
            registry_rows_added=1,
            cost_of_use_if_measurable=None,
            coverage_flags={"family_available": True},
        )
    else:
        family_name = "unknown"
        contribution = build_contribution_record(
            module_name="stage41_family_contribution_adapter",
            family_name=family_name,
            setup_name="fallback_setup",
            raw_candidate_contribution=0.0,
            stage_a_survival_lift=0.0,
            stage_b_survival_lift=0.0,
            final_policy_contribution=0.0,
            runtime_seconds=0.0,
            registry_rows_added=0,
            cost_of_use_if_measurable=None,
            coverage_flags={"family_available": False},
        )

    failure = build_failure_record(
        module_name="stage42_failure_adapter",
        family_name=family_name,
        motif="REJECT::NO_SIGNAL",
        details={"source": "stage42", "motif_examples": (stage42.get("family_memory", {}) or {}).get("recurring_failure_motifs", {})},
    )
    runtime = build_runtime_event(
        module_name="stage44_contract_probe",
        phase_name="adapter_probe",
        enter_ts=10.0,
        exit_ts=10.004,
        candidate_rows_in=5,
        candidate_rows_out=3,
    )
    allocator = build_allocator_hook(
        module_name="stage44_contract_probe",
        family_name=family_name,
        exploration_eligible=True,
        exploitation_score=0.45,
        uncertainty_score=0.55,
        novelty_score=0.30,
        min_exploration_floor=0.10,
    )
    registry_row = to_registry_row(
        module_name="stage44_contract_probe",
        family_name=family_name,
        setup_name="adapter_probe_setup",
        contribution_summary=contribution,
        failure_motifs=[str(failure.get("motif", "REJECT::NO_SIGNAL"))],
        runtime_metrics=runtime,
        allocator_hook=allocator,
        mutation_guidance="preserve_flow_and_reduce_cost_drag",
    )

    remaining_gaps: list[str] = []
    notes: list[str] = []
    if not contrib_rows:
        remaining_gaps.append("No Stage-41 family contributions found; fallback adapter used.")
        notes.append("Stage-44 contract probe completed with fallback contribution record.")

    status = "SUCCESS" if not remaining_gaps else "PARTIAL"
    modules_covered = sorted(
        {
            "stage41_family_contribution_adapter",
            "stage42_failure_adapter",
            "stage44_contract_probe",
            str(registry_row.get("family_name", "")),
        }
    )
    payload = {
        "stage": "44",
        "status": status,
        "contribution_contract_defined": True,
        "failure_contract_defined": True,
        "runtime_contract_defined": True,
        "allocator_hooks_defined": True,
        "registry_compatibility_defined": True,
        "modules_covered": modules_covered,
        "remaining_gaps": remaining_gaps,
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "status": payload["status"],
            "modules_covered": payload["modules_covered"],
            "remaining_gaps": payload["remaining_gaps"],
            "contribution": contribution,
            "failure": failure,
            "runtime": runtime,
            "allocator": allocator,
        },
        length=16,
    )
    validate_stage44_summary(payload)

    summary_path = docs_dir / "stage44_optimization_framework_summary.json"
    report_path = docs_dir / "stage44_optimization_framework_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(payload, notes=notes), encoding="utf-8")

    print(f"status: {payload['status']}")
    print(f"summary_hash: {payload['summary_hash']}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

