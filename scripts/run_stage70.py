"""Run Stage-70 search expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage51 import resolve_budget_mode, resolve_research_scope
from buffmini.stage70 import deduplicate_economic_candidates, generate_expanded_candidates
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-70 search expansion")
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
    candidates = generate_expanded_candidates(
        discovery_timeframes=[str(v) for v in scope["discovery_timeframes"]],
        budget_mode_selected=str(budget["selected"]),
    )
    candidates = deduplicate_economic_candidates(candidates)
    candidates.to_csv(docs_dir / "stage70_expanded_candidates.csv", index=False)
    target = 10000 if str(budget["selected"]) == "full_audit" else 2500
    family_count = int(candidates["family"].nunique()) if not candidates.empty and "family" in candidates.columns else 0
    timeframe_count = int(candidates["timeframe"].nunique()) if not candidates.empty and "timeframe" in candidates.columns else 0
    fp_count = int(candidates["economic_fingerprint"].nunique()) if not candidates.empty and "economic_fingerprint" in candidates.columns else 0
    diversity_ok = bool(family_count >= 8 and timeframe_count >= 2 and fp_count >= int(target * 0.90))
    status = "SUCCESS" if int(len(candidates)) >= target and diversity_ok else "PARTIAL"
    summary = {
        "stage": "70",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "heuristic_filter",
        "validation_state": "CANDIDATE_GENERATION_PASSED" if status == "SUCCESS" else "CANDIDATE_GENERATION_WEAK",
        "budget_mode_selected": str(budget["selected"]),
        "candidate_count": int(len(candidates)),
        "family_count": family_count,
        "timeframe_count": timeframe_count,
        "economic_fingerprint_count": fp_count,
        "diversity_ok": diversity_ok,
        "target_min_candidates": int(target),
        "blocker_reason": "" if status == "SUCCESS" else ("candidate_count_or_diversity_below_target"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage70_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage70_report.md").write_text(
        "\n".join(
            [
                "# Stage-70 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- budget_mode_selected: `{summary['budget_mode_selected']}`",
                f"- candidate_count: `{summary['candidate_count']}`",
                f"- family_count: `{summary['family_count']}`",
                f"- timeframe_count: `{summary['timeframe_count']}`",
                f"- economic_fingerprint_count: `{summary['economic_fingerprint_count']}`",
                f"- diversity_ok: `{summary['diversity_ok']}`",
                f"- target_min_candidates: `{summary['target_min_candidates']}`",
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
