"""Run Stage-87 mechanism and family refinement diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.mechanisms import generate_mechanism_source_candidates, mechanism_registry
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.stage51.scope import resolve_research_scope
from buffmini.stage52 import build_setup_candidate_v2
from buffmini.stage70.search_expansion import collapse_similarity_candidates
from buffmini.utils.hashing import stable_hash


SCHEMA_FIELDS = [
    "family",
    "subfamily",
    "context",
    "trigger",
    "confirmation",
    "participation",
    "invalidation",
    "risk_model",
    "exit_family",
    "time_stop_bars",
    "expected_regime",
    "expected_failure_modes",
    "trade_density_expectation",
    "transfer_expectation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-87 mechanism and family refinement diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    scope = resolve_research_scope(cfg)
    raw = generate_mechanism_source_candidates(
        discovery_timeframes=list(scope.get("discovery_timeframes", [])),
        budget_mode_selected=str((cfg.get("budget_mode", {}) or {}).get("selected", "search")),
        active_families=list(scope.get("active_setup_families", [])),
        target_min_candidates=800,
    )
    collapsed = collapse_similarity_candidates(raw, max_per_bucket=3)
    upgraded = build_setup_candidate_v2(dict(raw.iloc[0].to_dict()), timeframe=str(raw.iloc[0]["timeframe"])) if not raw.empty else {}
    registry = mechanism_registry()
    summary = {
        "stage": "87",
        "status": "SUCCESS" if not raw.empty else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "heuristic_filter",
        "validation_state": "MECHANISM_REFINEMENT_READY" if not raw.empty else "MECHANISM_REFINEMENT_INCOMPLETE",
        "registry_family_count": int(len(registry)),
        "raw_candidate_count": int(len(raw)),
        "post_similarity_collapse_count": int(len(collapsed)),
        "trivial_redundancy_reduction": float(round((1.0 - (len(collapsed) / max(1, len(raw)))) if len(raw) else 0.0, 6)),
        "candidate_schema_supported_fields": [field for field in SCHEMA_FIELDS if field in upgraded],
        "sample_upgraded_candidate": {field: upgraded.get(field) for field in SCHEMA_FIELDS},
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-87 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- registry_family_count: `{summary['registry_family_count']}`",
        f"- raw_candidate_count: `{summary['raw_candidate_count']}`",
        f"- post_similarity_collapse_count: `{summary['post_similarity_collapse_count']}`",
        f"- trivial_redundancy_reduction: `{summary['trivial_redundancy_reduction']}`",
        f"- validation_state: `{summary['validation_state']}`",
    ]
    lines.extend([""] + markdown_rows("Mechanism Registry", registry, limit=8))
    lines.extend([""] + markdown_rows("Sample Upgraded Candidate", [summary["sample_upgraded_candidate"]], limit=1))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="87", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
