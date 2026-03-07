"""Master orchestrator for Stage-44 through Stage-50."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage50.reporting import validate_stage44_50_master_summary
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-44..50 master flow")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--budget-small", action="store_true")
    return parser.parse_args()


def _run_checked(cmd: list[str], *, label: str) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if int(proc.returncode) != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-80:])
        raise RuntimeError(f"{label} failed (exit={proc.returncode})\n{tail}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _final_verdict(*, stage47: dict[str, Any], stage48: dict[str, Any], stage49: dict[str, Any], stage50: dict[str, Any]) -> str:
    before = int(stage47.get("baseline_raw_candidate_count", 0))
    after = int(stage47.get("upgraded_raw_candidate_count", 0))
    stage_a = int(stage48.get("stage_a_survivors_after", 0))
    stage_b = int(stage48.get("stage_b_survivors_after", 0))
    registry = int(stage49.get("registry_rows", 0))
    live_after = float(stage50.get("live_best_exp_lcb_after", 0.0))
    upgraded_raw = int(stage50.get("upgraded_raw_signals", 0))
    if after > before and upgraded_raw <= 0 and live_after <= 0.0:
        return "RAW_SIGNAL_IMPROVED_BUT_COST_STILL_DOMINANT"
    if stage_a > 0 and stage_b > 0 and live_after <= 0.0:
        return "TRADABILITY_IMPROVED_BUT_NO_ROBUST_EDGE"
    if registry > 0 and live_after <= 0.0:
        return "SELF_LEARNING_BECOMING_USEFUL_BUT_NOT_MATURE"
    if after > before:
        return "PARTIAL_PROGRESS_NEEDS_MORE_SETUP_POWER"
    return "NO_MEANINGFUL_PROGRESS"


def _render(payload: dict[str, Any], *, stage44: dict[str, Any], stage45: dict[str, Any], stage46: dict[str, Any], stage47: dict[str, Any], stage48: dict[str, Any], stage49: dict[str, Any], stage50: dict[str, Any], seed50: dict[str, Any]) -> str:
    lines = [
        "# Stage-44..50 Master Report",
        "",
        "## Stage Status",
        f"- Stage-44: `{payload.get('stage44_status', '')}`",
        f"- Stage-45: `{payload.get('stage45_status', '')}`",
        f"- Stage-46: `{payload.get('stage46_status', '')}`",
        f"- Stage-47: `{payload.get('stage47_status', '')}`",
        f"- Stage-48: `{payload.get('stage48_status', '')}`",
        f"- Stage-49: `{payload.get('stage49_status', '')}`",
        f"- Stage-50: `{payload.get('stage50_status', '')}`",
        "",
        "## Optimization Framework + Analyst Brain",
        f"- Stage-44 modules_covered: `{stage44.get('modules_covered', [])}`",
        f"- Stage-45 modules_contract_compliant: `{bool(stage45.get('modules_contract_compliant', False))}`",
        f"- Stage-46 modules_contract_compliant: `{bool(stage46.get('modules_contract_compliant', False))}`",
        f"- OI short-only verified: `{bool(stage46.get('oi_short_only_guard_verified', False))}`",
        "",
        "## Setup and Tradability",
        f"- Stage-47 raw candidates before/after: `{payload.get('stage47_raw_candidates_before', 0)} -> {payload.get('stage47_raw_candidates_after', 0)}`",
        f"- Stage-48 Stage-A survivors: `{payload.get('stage48_stage_a_survivors', 0)}`",
        f"- Stage-48 Stage-B survivors: `{payload.get('stage48_stage_b_survivors', 0)}`",
        "",
        "## Self-Learning 3.0",
        f"- registry_rows: `{payload.get('stage49_registry_rows', 0)}`",
        f"- elites_count: `{payload.get('stage49_elites_count', 0)}`",
        f"- mutate_next: `{stage49.get('mutate_next', '')}`",
        "",
        "## Runtime and Validation",
        f"- baseline_runtime_seconds: `{payload.get('stage50_runtime_seconds_baseline', 0.0)}`",
        f"- upgraded_runtime_seconds: `{payload.get('stage50_runtime_seconds_upgraded', 0.0)}`",
        f"- stage50_promising: `{payload.get('stage50_promising', False)}`",
        f"- stage50_5seed_executed: `{payload.get('stage50_5seed_executed', 0)}`",
        f"- stage50_5seed_skipped: `{bool(seed50.get('skipped', False))}`",
        f"- stage50_5seed_skip_reason: `{seed50.get('skip_reason_if_any', '')}`",
        "",
        "## Final Verdict",
        f"- final_verdict: `{payload.get('final_verdict', '')}`",
        f"- biggest_remaining_bottleneck: `{payload.get('biggest_remaining_bottleneck', '')}`",
        f"- next_cheapest_action: `{payload.get('next_cheapest_action', '')}`",
        f"- deterministic_summary_hash: `{payload.get('deterministic_summary_hash', '')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    _run_checked([sys.executable, "scripts/run_stage44.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir)], label="stage44")
    _run_checked([sys.executable, "scripts/run_stage45.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir), "--config", str(args.config)], label="stage45")
    _run_checked([sys.executable, "scripts/run_stage46.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir), "--config", str(args.config)], label="stage46")
    _run_checked([sys.executable, "scripts/run_stage47.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir)], label="stage47")
    _run_checked([sys.executable, "scripts/run_stage48.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir), "--config", str(args.config)], label="stage48")
    _run_checked([sys.executable, "scripts/run_stage49.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir)], label="stage49")
    cmd50 = [sys.executable, "scripts/run_stage50.py", "--seed", str(int(args.seed)), "--docs-dir", str(docs_dir), "--config", str(args.config)]
    if bool(args.budget_small):
        cmd50.append("--budget-small")
    _run_checked(cmd50, label="stage50")

    stage44 = _load_json(docs_dir / "stage44_optimization_framework_summary.json")
    stage45 = _load_json(docs_dir / "stage45_analyst_brain_part1_summary.json")
    stage46 = _load_json(docs_dir / "stage46_analyst_brain_part2_summary.json")
    stage47 = _load_json(docs_dir / "stage47_signal_gen2_summary.json")
    stage48 = _load_json(docs_dir / "stage48_tradability_learning_summary.json")
    stage49 = _load_json(docs_dir / "stage49_self_learning3_summary.json")
    stage50 = _load_json(docs_dir / "stage50_performance_validation_summary.json")
    seed50 = _load_json(docs_dir / "stage50_5seed_summary.json")

    final_verdict = _final_verdict(stage47=stage47, stage48=stage48, stage49=stage49, stage50=stage50)
    biggest_bottleneck = "cost_drag_vs_signal" if int(stage50.get("upgraded_raw_signals", 0)) <= 0 else "robustness_gap"
    next_action = "Increase setup quality per trade (RR and cost-adjusted edge) and rerun Stage-48/50 with tightened cost-drag motifs."

    payload = {
        "stage44_status": str(stage44.get("status", "PARTIAL")),
        "stage45_status": str(stage45.get("status", "PARTIAL")),
        "stage46_status": str(stage46.get("status", "PARTIAL")),
        "stage47_status": str(stage47.get("status", "PARTIAL")),
        "stage48_status": str(stage48.get("status", "PARTIAL")),
        "stage49_status": str(stage49.get("status", "PARTIAL")),
        "stage50_status": str(stage50.get("status", "PARTIAL")),
        "stage47_raw_candidates_before": int(stage47.get("baseline_raw_candidate_count", 0)),
        "stage47_raw_candidates_after": int(stage47.get("upgraded_raw_candidate_count", 0)),
        "stage48_stage_a_survivors": int(stage48.get("stage_a_survivors_after", 0)),
        "stage48_stage_b_survivors": int(stage48.get("stage_b_survivors_after", 0)),
        "stage49_registry_rows": int(stage49.get("registry_rows", 0)),
        "stage49_elites_count": int(stage49.get("elites_count", 0)),
        "stage50_runtime_seconds_baseline": float(stage50.get("baseline_runtime_seconds", 0.0)),
        "stage50_runtime_seconds_upgraded": float(stage50.get("upgraded_runtime_seconds", 0.0)),
        "stage50_promising": bool(stage50.get("promising", False)),
        "stage50_5seed_executed": int(len(seed50.get("executed_seeds", []))),
        "final_verdict": final_verdict,
        "biggest_remaining_bottleneck": biggest_bottleneck,
        "next_cheapest_action": next_action,
    }
    payload["deterministic_summary_hash"] = stable_hash(
        {
            "stage44_status": payload["stage44_status"],
            "stage45_status": payload["stage45_status"],
            "stage46_status": payload["stage46_status"],
            "stage47_status": payload["stage47_status"],
            "stage48_status": payload["stage48_status"],
            "stage49_status": payload["stage49_status"],
            "stage50_status": payload["stage50_status"],
            "stage47_raw_candidates_before": payload["stage47_raw_candidates_before"],
            "stage47_raw_candidates_after": payload["stage47_raw_candidates_after"],
            "stage48_stage_a_survivors": payload["stage48_stage_a_survivors"],
            "stage48_stage_b_survivors": payload["stage48_stage_b_survivors"],
            "stage49_registry_rows": payload["stage49_registry_rows"],
            "stage49_elites_count": payload["stage49_elites_count"],
            "stage50_runtime_seconds_baseline": payload["stage50_runtime_seconds_baseline"],
            "stage50_runtime_seconds_upgraded": payload["stage50_runtime_seconds_upgraded"],
            "stage50_promising": payload["stage50_promising"],
            "stage50_5seed_executed": payload["stage50_5seed_executed"],
            "final_verdict": payload["final_verdict"],
            "biggest_remaining_bottleneck": payload["biggest_remaining_bottleneck"],
            "next_cheapest_action": payload["next_cheapest_action"],
        },
        length=16,
    )
    validate_stage44_50_master_summary(payload)

    master_summary = docs_dir / "stage44_50_master_summary.json"
    master_report = docs_dir / "stage44_50_master_report.md"
    master_summary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    master_report.write_text(
        _render(
            payload,
            stage44=stage44,
            stage45=stage45,
            stage46=stage46,
            stage47=stage47,
            stage48=stage48,
            stage49=stage49,
            stage50=stage50,
            seed50=seed50,
        ),
        encoding="utf-8",
    )

    print(f"deterministic_summary_hash: {payload['deterministic_summary_hash']}")
    print(f"master_report: {master_report}")
    print(f"master_summary: {master_summary}")


if __name__ == "__main__":
    main()

