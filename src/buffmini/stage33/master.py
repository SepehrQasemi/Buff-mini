"""Master summary builders for Stage-30..33 reporting."""

from __future__ import annotations

from typing import Any


def choose_verdict(
    *,
    stage32_metrics: dict[str, Any],
    stage33_metrics: dict[str, Any],
) -> str:
    wf = float(stage32_metrics.get("wf_executed_pct", 0.0))
    mc = float(stage32_metrics.get("mc_trigger_pct", 0.0))
    exp_live = float((stage33_metrics.get("policy_metrics", {}) or {}).get("live", {}).get("exp_lcb", 0.0))
    if wf <= 0.0 or mc <= 0.0:
        return "INSUFFICIENT_DATA"
    if exp_live > 0.02 and wf >= 50.0 and mc >= 20.0:
        return "ROBUST_EDGE"
    if exp_live > 0.0:
        return "CONTEXTUAL_EDGE"
    if float((stage33_metrics.get("policy_metrics", {}) or {}).get("research", {}).get("exp_lcb", 0.0)) > 0.0:
        return "WEAK_EDGE"
    return "NO_EDGE"


def build_master_summary(
    *,
    head_commit: str,
    run_ids: dict[str, str],
    stage30: dict[str, Any],
    stage31: dict[str, Any],
    stage32: dict[str, Any],
    stage33: dict[str, Any],
    drift: dict[str, Any],
    config_hash: str,
    data_hash: str,
    resolved_end_ts: str | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    verdict = choose_verdict(stage32_metrics=stage32, stage33_metrics=stage33)
    next_bottleneck = (
        "wf_mc_preconditions"
        if float(stage32.get("wf_executed_pct", 0.0)) <= 0.0 or float(stage32.get("mc_trigger_pct", 0.0)) <= 0.0
        else "cost_drag_vs_signal"
        if float((stage33.get("policy_metrics", {}) or {}).get("live", {}).get("exp_lcb", 0.0)) <= 0.0
        else "none"
    )
    return {
        "stage": "30_33_master",
        "head_commit": str(head_commit),
        "run_ids": dict(run_ids),
        "config_hash": str(config_hash),
        "data_hash": str(data_hash),
        "resolved_end_ts": resolved_end_ts,
        "stage30": dict(stage30),
        "stage31": dict(stage31),
        "stage32": dict(stage32),
        "stage33": dict(stage33),
        "drift": dict(drift),
        "verdict": verdict,
        "next_bottleneck": next_bottleneck,
        "runtime_seconds": float(runtime_seconds),
    }


def render_master_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-30 to Stage-33 Master Report",
        "",
        f"- head_commit: `{summary.get('head_commit', '')}`",
        f"- verdict: `{summary.get('verdict', '')}`",
        f"- next_bottleneck: `{summary.get('next_bottleneck', '')}`",
        "",
        "## Stage Summaries",
        f"- stage30: `{summary.get('stage30', {})}`",
        f"- stage31: `{summary.get('stage31', {})}`",
        f"- stage32: `{summary.get('stage32', {})}`",
        f"- stage33: `{summary.get('stage33', {})}`",
        "",
        "## Drift",
        f"- drift: `{summary.get('drift', {})}`",
        "",
        "## Next Actions",
    ]
    verdict = str(summary.get("verdict", ""))
    if verdict in {"NO_EDGE", "WEAK_EDGE"}:
        lines.extend(
            [
                "1. Expand contextual feature space with new free-data transforms (market structure + session microstructure).",
                "2. Increase search budget on a server profile while preserving exploration quotas and diversity constraints.",
                "3. Tighten feasibility-aware policy activation so only contexts with positive rolling evidence are tradable live.",
            ]
        )
    else:
        lines.extend(
            [
                "1. Freeze policy snapshot and run forward paper monitoring.",
                "2. Expand symbols/timeframes under the same validation gates.",
                "3. Track drift daily and trigger retraining when representation/performance thresholds break.",
            ]
        )
    return "\n".join(lines).strip() + "\n"
