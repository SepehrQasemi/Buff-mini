"""Run Stage-59 conditional expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage51 import resolve_research_scope
from buffmini.stage59 import build_conditional_expansion
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-59 conditional expansion")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    scope = resolve_research_scope(cfg)
    stage58_path = docs_dir / "stage58_summary.json"
    stage58 = json.loads(stage58_path.read_text(encoding="utf-8")) if stage58_path.exists() else {"transfer_result": {"verdict": "NO_EDGE_IN_SCOPE"}}
    expansion = build_conditional_expansion(
        stage58_verdict=str(stage58.get("transfer_result", {}).get("verdict", "NO_EDGE_IN_SCOPE")),
        transfer_acceptable=bool(stage58.get("transfer_result", {}).get("transfer_acceptable", False)),
        active_families=[str(v) for v in scope["active_setup_families"]],
        next_families=["crowded_side_squeeze", "flow_exhaustion_reversal", "regime_shift_entry"],
        oi_short_horizon_only=True,
        paid_provider_optional=True,
    )
    status = "SUCCESS" if bool(expansion.get("expansion_allowed", False)) else "PARTIAL"
    execution_status = "EXECUTED"
    validation_state = "EXPANSION_APPROVED" if bool(expansion.get("expansion_allowed", False)) else "EXPANSION_BLOCKED"
    summary = {
        "stage": "59",
        "status": status,
        "execution_status": execution_status,
        "stage_role": "reporting_only",
        "validation_state": validation_state,
        "stage58_verdict": str(stage58.get("transfer_result", {}).get("verdict", "NO_EDGE_IN_SCOPE")),
        "expansion_plan": expansion,
        "summary_hash": stable_hash(
            {
                "status": status,
                "execution_status": execution_status,
                "stage_role": "reporting_only",
                "validation_state": validation_state,
                "stage58_verdict": str(stage58.get("transfer_result", {}).get("verdict", "NO_EDGE_IN_SCOPE")),
                "expansion_plan": expansion,
            },
            length=16,
        ),
    }
    summary_path = docs_dir / "stage59_summary.json"
    report_path = docs_dir / "stage59_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-59 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage58_verdict: `{summary['stage58_verdict']}`",
                f"- expansion_plan: `{summary['expansion_plan']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
