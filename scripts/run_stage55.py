"""Run Stage-55 replay efficiency using Stage-54 artifacts and optional runtime probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage51 import resolve_budget_mode
from buffmini.stage55 import allocate_replay_budget, estimate_replay_speedup, validate_phase_timings
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-55 replay efficiency")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage54 = _load_json(docs_dir / "stage54_summary.json")
    if str(stage54.get("stage28_run_id", "")).strip():
        return str(stage54["stage28_run_id"]).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    stage54_summary = _load_json(docs_dir / "stage54_summary.json")

    budget_info = resolve_budget_mode(cfg)
    budget = dict(budget_info["active"])

    candidates = pd.DataFrame()
    if stage28_run_id:
        pruned_path = Path(args.runs_dir) / stage28_run_id / "stage54" / "pruned_candidates.csv"
        if pruned_path.exists():
            candidates = pd.read_csv(pruned_path)
    if candidates.empty:
        candidates = pd.DataFrame(
            [
                {"candidate_id": f"s55_{idx}", "replay_priority": round(0.90 - idx * 0.05, 6)}
                for idx in range(24)
            ]
        )
        input_mode = "bootstrap_candidates"
    else:
        input_mode = "stage54_pruned_candidates"
        if "replay_priority" not in candidates.columns:
            candidates["replay_priority"] = 0.0
        if "candidate_id" not in candidates.columns:
            candidates["candidate_id"] = [f"s55_auto_{idx}" for idx in range(len(candidates))]

    allocation = allocate_replay_budget(candidates, budget=budget)

    stage43 = _load_json(docs_dir / "stage43_performance_summary.json")
    phase_runtime = dict(stage43.get("phase_runtime_seconds", {}))
    phase_timings = {
        "candidate_generation": float(phase_runtime.get("candidate_generation", 0.5)),
        "stage_a_gate": float(phase_runtime.get("stage_a_objective", 0.4)),
        "stage_b_gate": float(phase_runtime.get("stage_b_objective", 0.3)),
        "micro_replay": float(phase_runtime.get("replay_backtest", 0.0)) * 0.25,
        "full_replay": float(phase_runtime.get("replay_backtest", 0.0)) * 0.75,
        "walkforward": float(phase_runtime.get("walkforward", 0.0)),
        "monte_carlo": float(phase_runtime.get("monte_carlo", 0.0)),
    }
    validate_phase_timings(phase_timings)

    baseline_runtime = float(stage43.get("baseline", {}).get("runtime_seconds", 100.0))
    upgraded_runtime = float(stage43.get("upgraded", {}).get("runtime_seconds", baseline_runtime))
    full_replay_cap = int(allocation["counts"]["full_replay"])
    precheck_cap = int(max(1, allocation["counts"]["precheck"]))
    runtime_scale = max(0.25, min(1.0, full_replay_cap / precheck_cap))
    optimized_runtime = upgraded_runtime * runtime_scale
    speedup_raw = estimate_replay_speedup(
        baseline_runtime_seconds=baseline_runtime,
        optimized_runtime_seconds=optimized_runtime,
    )
    speedup_projection = {
        "baseline_runtime_seconds": float(speedup_raw["baseline_runtime_seconds"]),
        "optimized_runtime_seconds": float(speedup_raw["optimized_runtime_seconds"]),
        "improvement_pct": float(speedup_raw["improvement_pct"]),
        "projected_meets_target": bool(speedup_raw["meets_stage55_target"]),
    }
    dead_path = str(stage54_summary.get("status", "SUCCESS")) != "SUCCESS"
    measured_probe = {}
    if stage28_run_id:
        measured_probe = _load_json(Path(args.runs_dir) / stage28_run_id / "stage55" / "phase_timings_measured.json")
    measured_timings = dict(measured_probe.get("phase_timings", {})) if isinstance(measured_probe, dict) else {}
    measured_runtime_seconds = 0.0
    measured_improvement_pct = 0.0
    measured_enabled = False
    measured_source = "disabled_due_to_dead_path" if dead_path else "no_runtime_probe"
    if not dead_path and measured_timings:
        validate_phase_timings(measured_timings)
        measured_runtime_seconds = float(sum(float(v) for v in measured_timings.values()))
        measured_improvement_pct = float(((baseline_runtime - measured_runtime_seconds) / max(baseline_runtime, 1e-9)) * 100.0)
        measured_enabled = bool(measured_runtime_seconds > 0.0)
        measured_source = "stage55_runtime_probe"
    speedup_measurement = {
        "measured": bool(measured_enabled),
        "source": measured_source,
        "measured_runtime_seconds": float(round(measured_runtime_seconds, 6)) if measured_enabled else 0.0,
        "measured_improvement_pct": float(round(measured_improvement_pct, 6)) if measured_enabled else 0.0,
        "meets_stage55_target": bool(measured_enabled and measured_improvement_pct >= 40.0),
    }
    projection_only = bool(not measured_enabled)
    status = "SUCCESS" if (not dead_path and not projection_only) else "PARTIAL"
    summary = {
        "stage": "55",
        "status": status,
        "execution_status": "EXECUTED",
        "validation_state": "MEASURED_PERFORMANCE" if status == "SUCCESS" else "PROJECTED_ONLY_OR_BLOCKED",
        "input_mode": input_mode,
        "stage28_run_id": stage28_run_id,
        "selected_budget_mode": budget_info["selected"],
        "projection_only": projection_only,
        "allocation_counts": allocation["counts"],
        "phase_timings": phase_timings,
        "speedup_projection": speedup_projection,
        "speedup_measurement": speedup_measurement,
        "blocker_reason": (
            "upstream_stage54_not_success"
            if dead_path
            else ("performance_not_measured_runtime_probe_missing" if projection_only else "")
        ),
        "summary_hash": stable_hash(
            {
                "status": status,
                "validation_state": "MEASURED_PERFORMANCE" if status == "SUCCESS" else "PROJECTED_ONLY_OR_BLOCKED",
                "input_mode": input_mode,
                "stage28_run_id": stage28_run_id,
                "selected_budget_mode": budget_info["selected"],
                "allocation_counts": allocation["counts"],
                "speedup_projection": speedup_projection,
                "speedup_measurement": speedup_measurement,
                "blocker_reason": (
                    "upstream_stage54_not_success"
                    if dead_path
                    else ("performance_not_measured_runtime_probe_missing" if projection_only else "")
                ),
            },
            length=16,
        ),
    }

    if stage28_run_id:
        out_dir = Path(args.runs_dir) / stage28_run_id / "stage55"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "allocation.json").write_text(json.dumps(allocation, indent=2, allow_nan=False), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage55_summary.json"
    report_path = docs_dir / "stage55_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-55 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- input_mode: `{summary['input_mode']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- selected_budget_mode: `{summary['selected_budget_mode']}`",
                f"- projection_only: `{summary['projection_only']}`",
                f"- allocation_counts: `{summary['allocation_counts']}`",
                f"- speedup_projection: `{summary['speedup_projection']}`",
                f"- speedup_measurement: `{summary['speedup_measurement']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
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
