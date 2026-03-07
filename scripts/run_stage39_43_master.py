"""Master orchestrator for Stage-39 through Stage-43."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-39..43 master flow")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _run_checked(cmd: list[str], *, label: str) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    if int(proc.returncode) != 0:
        tail = "\n".join((stdout + "\n" + stderr).splitlines()[-80:])
        raise RuntimeError(f"{label} failed (exit={proc.returncode})\n{tail}")
    return stdout


def _load_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    if str(stage39.get("stage28_run_id", "")).strip():
        return str(stage39["stage28_run_id"]).strip()
    stage38 = _load_json(docs_dir / "stage38_master_summary.json")
    return str(stage38.get("stage28_run_id", "")).strip()


def _master_verdict(
    *,
    stage39: dict[str, Any],
    stage40: dict[str, Any],
    stage41: dict[str, Any],
    stage42: dict[str, Any],
    stage43_perf: dict[str, Any],
) -> str:
    baseline_raw = int(stage39.get("baseline_engine_raw_signal_count", 0))
    upgraded_raw_candidates = int(stage39.get("raw_candidate_count", 0))
    stage_a = int((stage40.get("counts", {}) or {}).get("stage_a", 0))
    stage_b = int((stage40.get("counts", {}) or {}).get("stage_b", 0))
    upgraded_trade_count = float((stage43_perf.get("upgraded", {}) or {}).get("trade_count", 0.0))
    upgraded_live_lcb = float((stage43_perf.get("upgraded", {}) or {}).get("live_best_exp_lcb", 0.0))
    delta_activation = float((stage43_perf.get("delta", {}) or {}).get("delta_activation_rate", 0.0))
    registry_rows = int(stage42.get("registry_rows_v2", 0))
    active_families = sum(
        1
        for row in stage41.get("family_contributions", [])
        if float(row.get("stage_b_count", 0.0)) > 0.0
    )

    if upgraded_raw_candidates > baseline_raw and upgraded_trade_count <= 0.0 and upgraded_live_lcb <= 0.0:
        return "RAW_SIGNAL_IMPROVED_BUT_NO_EDGE"
    if stage_a > 0 and stage_b > 0 and delta_activation > 0.0 and upgraded_live_lcb <= 0.0:
        return "TRADABILITY_IMPROVED_BUT_ROBUSTNESS_WEAK"
    if registry_rows > 0 and upgraded_trade_count <= 0.0:
        return "SELF_LEARNING_IMPROVED_BUT_SIGNAL_STILL_WEAK"
    if active_families >= 2 and upgraded_trade_count <= 0.0:
        return "PARTIAL_PROGRESS_NEEDS_MORE_SIGNAL_FAMILIES"
    return "NO_MEANINGFUL_PROGRESS"


def _render_master(payload: dict[str, Any]) -> str:
    stage39 = dict(payload.get("stage39", {}))
    stage40 = dict(payload.get("stage40", {}))
    stage41 = dict(payload.get("stage41", {}))
    stage42 = dict(payload.get("stage42", {}))
    stage43_perf = dict(payload.get("stage43_performance", {}))
    stage43_seed = dict(payload.get("stage43_5seed", {}))
    lines = [
        "# Stage-39..43 Master Report",
        "",
        "## Stage Changes",
        "- Stage-39: widened context grammar and layered candidate funnel (A/B/C).",
        "- Stage-40: tradability labels + two-stage objective routing.",
        "- Stage-41: derivatives family contribution metrics with OI short-only runtime guard.",
        "- Stage-42: failure-aware self-learning memory, motifs, and mutation guidance.",
        "- Stage-43: runtime instrumentation, baseline-vs-upgraded comparison, and 5-seed validation gate.",
        "",
        "## Upstream Signal Generation",
        f"- baseline_engine_raw_signal_count: `{int(stage39.get('baseline_engine_raw_signal_count', 0))}`",
        f"- upgraded_raw_candidate_count: `{int(stage39.get('raw_candidate_count', 0))}`",
        f"- shortlisted_count: `{int(stage39.get('shortlisted_count', 0))}`",
        "",
        "## Tradability Objective",
        f"- stage_a_survivors: `{int((stage40.get('counts', {}) or {}).get('stage_a', 0))}`",
        f"- stage_b_survivors: `{int((stage40.get('counts', {}) or {}).get('stage_b', 0))}`",
        f"- strongest_bottleneck_step: `{stage40.get('strongest_bottleneck_step', '')}`",
        "",
        "## Derivatives Contributions",
        f"- funding_available: `{bool(stage41.get('funding_available', False))}`",
        f"- taker_available: `{bool(stage41.get('taker_available', False))}`",
        f"- long_short_available: `{bool(stage41.get('long_short_available', False))}`",
        f"- oi_short_only_mode_enabled: `{bool(stage41.get('oi_short_only_mode_enabled', False))}`",
        "",
        "## Self-Learning 2.0",
        f"- registry_rows_v2: `{int(stage42.get('registry_rows_v2', 0))}`",
        f"- elite_count: `{int(stage42.get('elite_count', 0))}`",
        f"- mutate_next: `{((stage42.get('self_diagnosis', {}) or {}).get('mutate_next', ''))}`",
        "",
        "## Engine Performance (Baseline vs Upgraded)",
        f"- baseline_run_id: `{((stage43_perf.get('baseline', {}) or {}).get('run_id', ''))}`",
        f"- upgraded_run_id: `{((stage43_perf.get('upgraded', {}) or {}).get('run_id', ''))}`",
        f"- delta_raw_signal_count: `{float((stage43_perf.get('delta', {}) or {}).get('delta_raw_signal_count', 0.0)):.6f}`",
        f"- delta_activation_rate: `{float((stage43_perf.get('delta', {}) or {}).get('delta_activation_rate', 0.0)):.6f}`",
        f"- delta_trade_count: `{float((stage43_perf.get('delta', {}) or {}).get('delta_trade_count', 0.0)):.6f}`",
        f"- delta_live_best_exp_lcb: `{float((stage43_perf.get('delta', {}) or {}).get('delta_live_best_exp_lcb', 0.0)):.6f}`",
        "",
        "## Runtime",
        f"- slowest_phase: `{stage43_perf.get('slowest_phase', '')}`",
        f"- replay_backtest_seconds: `{float(((stage43_perf.get('phase_runtime_seconds', {}) or {}).get('replay_backtest', 0.0))):.6f}`",
        "",
        "## 5-Seed Validation",
        f"- skipped: `{bool(stage43_seed.get('skipped', False))}`",
        f"- executed_seed_count: `{int(stage43_seed.get('executed_seed_count', 0))}`",
        f"- note: `{stage43_seed.get('note', '')}`",
        "",
        "## Final Verdict",
        f"- verdict: `{payload.get('verdict', '')}`",
        f"- biggest_remaining_bottleneck: `{payload.get('biggest_remaining_bottleneck', '')}`",
        f"- next_cheapest_high_confidence_action: `{payload.get('next_cheapest_action', '')}`",
        f"- summary_hash: `{payload.get('summary_hash', '')}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)

    cmd39 = [
        sys.executable,
        "scripts/run_stage39.py",
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if stage28_run_id:
        cmd39.extend(["--stage28-run-id", stage28_run_id])
    _run_checked(cmd39, label="run_stage39")

    cmd40 = [
        sys.executable,
        "scripts/run_stage40.py",
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if stage28_run_id:
        cmd40.extend(["--stage28-run-id", stage28_run_id])
    _run_checked(cmd40, label="run_stage40")

    cmd41 = [
        sys.executable,
        "scripts/run_stage41.py",
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if stage28_run_id:
        cmd41.extend(["--stage28-run-id", stage28_run_id])
    _run_checked(cmd41, label="run_stage41")

    cmd42 = [
        sys.executable,
        "scripts/run_stage42.py",
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if stage28_run_id:
        cmd42.extend(["--stage28-run-id", stage28_run_id])
    _run_checked(cmd42, label="run_stage42")

    cmd43 = [
        sys.executable,
        "scripts/run_stage43.py",
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seed)),
        "--runs-dir",
        str(runs_dir),
        "--docs-dir",
        str(docs_dir),
    ]
    if bool(args.budget_small):
        cmd43.append("--budget-small")
    _run_checked(cmd43, label="run_stage43")

    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    stage40 = _load_json(docs_dir / "stage40_tradability_objective_summary.json")
    stage41 = _load_json(docs_dir / "stage41_derivatives_completion_summary.json")
    stage42 = _load_json(docs_dir / "stage42_self_learning_2_summary.json")
    stage43_performance = _load_json(docs_dir / "stage43_performance_summary.json")
    stage43_5seed = _load_json(docs_dir / "stage43_5seed_summary.json")

    verdict = _master_verdict(
        stage39=stage39,
        stage40=stage40,
        stage41=stage41,
        stage42=stage42,
        stage43_perf=stage43_performance,
    )
    biggest_remaining_bottleneck = str((stage43_performance.get("upgraded", {}) or {}).get("next_bottleneck", "upstream_signal_generation"))
    next_action = "Increase upstream tradable signal families and tune Stage-A acceptance while preserving cost realism."

    payload = {
        "stage": "39_43",
        "seed": int(args.seed),
        "stage28_run_id": str(stage39.get("stage28_run_id", stage28_run_id)),
        "stage39": stage39,
        "stage40": stage40,
        "stage41": stage41,
        "stage42": stage42,
        "stage43_performance": stage43_performance,
        "stage43_5seed": stage43_5seed,
        "verdict": verdict,
        "biggest_remaining_bottleneck": biggest_remaining_bottleneck,
        "next_cheapest_action": next_action,
    }
    perf_stable = {
        "baseline": {
            "run_id": str((stage43_performance.get("baseline", {}) or {}).get("run_id", "")),
            "summary_hash": str((stage43_performance.get("baseline", {}) or {}).get("summary_hash", "")),
            "raw_signal_count": int((stage43_performance.get("baseline", {}) or {}).get("raw_signal_count", 0)),
            "activation_rate": float((stage43_performance.get("baseline", {}) or {}).get("activation_rate", 0.0)),
            "trade_count": float((stage43_performance.get("baseline", {}) or {}).get("trade_count", 0.0)),
            "live_best_exp_lcb": float((stage43_performance.get("baseline", {}) or {}).get("live_best_exp_lcb", 0.0)),
        },
        "upgraded": {
            "run_id": str((stage43_performance.get("upgraded", {}) or {}).get("run_id", "")),
            "summary_hash": str((stage43_performance.get("upgraded", {}) or {}).get("summary_hash", "")),
            "raw_signal_count": int((stage43_performance.get("upgraded", {}) or {}).get("raw_signal_count", 0)),
            "activation_rate": float((stage43_performance.get("upgraded", {}) or {}).get("activation_rate", 0.0)),
            "trade_count": float((stage43_performance.get("upgraded", {}) or {}).get("trade_count", 0.0)),
            "live_best_exp_lcb": float((stage43_performance.get("upgraded", {}) or {}).get("live_best_exp_lcb", 0.0)),
        },
        "delta": {
            "delta_raw_signal_count": float((stage43_performance.get("delta", {}) or {}).get("delta_raw_signal_count", 0.0)),
            "delta_activation_rate": float((stage43_performance.get("delta", {}) or {}).get("delta_activation_rate", 0.0)),
            "delta_trade_count": float((stage43_performance.get("delta", {}) or {}).get("delta_trade_count", 0.0)),
            "delta_live_best_exp_lcb": float((stage43_performance.get("delta", {}) or {}).get("delta_live_best_exp_lcb", 0.0)),
        },
        "promising": bool(stage43_performance.get("promising", False)),
    }
    seed_stable = {
        "skipped": bool(stage43_5seed.get("skipped", False)),
        "skip_reason": str(stage43_5seed.get("skip_reason", "")),
        "executed_seed_count": int(stage43_5seed.get("executed_seed_count", 0)),
        "distribution": dict(stage43_5seed.get("distribution", {})),
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "seed": payload["seed"],
            "stage28_run_id": payload["stage28_run_id"],
            "stage39": payload["stage39"],
            "stage40": payload["stage40"],
            "stage41": payload["stage41"],
            "stage42": payload["stage42"],
            "stage43_performance_stable": perf_stable,
            "stage43_5seed_stable": seed_stable,
            "verdict": payload["verdict"],
            "biggest_remaining_bottleneck": payload["biggest_remaining_bottleneck"],
        },
        length=16,
    )

    master_report = docs_dir / "stage39_43_master_report.md"
    master_summary = docs_dir / "stage39_43_master_summary.json"
    master_report.write_text(_render_master(payload), encoding="utf-8")
    master_summary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {payload['stage28_run_id']}")
    print(f"master_summary_hash: {payload['summary_hash']}")
    print(f"master_report: {master_report}")
    print(f"master_summary: {master_summary}")


if __name__ == "__main__":
    main()
