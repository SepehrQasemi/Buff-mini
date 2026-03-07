"""Stage-43 rendering and schema validation."""

from __future__ import annotations

from typing import Any

REQUIRED_PHASE_KEYS: tuple[str, ...] = (
    "config_load",
    "data_load",
    "extras_alignment",
    "feature_generation",
    "candidate_generation",
    "stage_a_objective",
    "stage_b_objective",
    "composer_policy_build",
    "replay_backtest",
    "walkforward",
    "monte_carlo",
    "report_generation",
)


def validate_stage43_performance_summary(payload: dict[str, Any]) -> None:
    """Validate Stage-43 performance summary schema."""

    required_top = {
        "stage",
        "seed",
        "baseline",
        "upgraded",
        "delta",
        "promising",
        "phase_runtime_seconds",
        "slowest_phase",
        "budget_mode",
        "summary_hash",
    }
    missing = sorted(required_top.difference(payload.keys()))
    if missing:
        raise ValueError(f"Missing Stage-43 performance keys: {missing}")
    if str(payload.get("stage", "")) != "43.3":
        raise ValueError("stage must be '43.3'")

    for side in ("baseline", "upgraded"):
        section = payload.get(side, {})
        if not isinstance(section, dict):
            raise ValueError(f"{side} must be object")
        for key in (
            "run_id",
            "summary_hash",
            "raw_signal_count",
            "activation_rate",
            "trade_count",
            "research_best_exp_lcb",
            "live_best_exp_lcb",
            "wf_executed_pct",
            "mc_trigger_pct",
            "runtime_seconds",
            "next_bottleneck",
        ):
            if key not in section:
                raise ValueError(f"{side}.{key} missing")

    delta = payload.get("delta", {})
    if not isinstance(delta, dict):
        raise ValueError("delta must be object")
    for key in (
        "delta_raw_signal_count",
        "delta_activation_rate",
        "delta_trade_count",
        "delta_research_best_exp_lcb",
        "delta_live_best_exp_lcb",
        "delta_runtime_seconds",
    ):
        if key not in delta:
            raise ValueError(f"delta.{key} missing")

    phases = payload.get("phase_runtime_seconds", {})
    if not isinstance(phases, dict):
        raise ValueError("phase_runtime_seconds must be object")
    for key in REQUIRED_PHASE_KEYS:
        if key not in phases:
            raise ValueError(f"phase_runtime_seconds.{key} missing")
        value = phases.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"phase_runtime_seconds.{key} must be numeric")
        if float(value) < 0.0:
            raise ValueError(f"phase_runtime_seconds.{key} must be >= 0")

    if not isinstance(payload.get("promising"), bool):
        raise ValueError("promising must be bool")
    if str(payload.get("slowest_phase", "")).strip() not in set(REQUIRED_PHASE_KEYS):
        raise ValueError("slowest_phase must be one of required phase keys")


def validate_stage43_5seed_summary(payload: dict[str, Any]) -> None:
    """Validate Stage-43 5-seed summary schema."""

    required_top = {
        "stage",
        "seed",
        "upgraded_reference_run_id",
        "executed_seed_count",
        "skipped",
        "skip_reason",
        "note",
        "rows",
        "distribution",
        "summary_hash",
    }
    missing = sorted(required_top.difference(payload.keys()))
    if missing:
        raise ValueError(f"Missing Stage-43 5-seed keys: {missing}")
    if str(payload.get("stage", "")) != "43.4":
        raise ValueError("stage must be '43.4'")
    if not isinstance(payload.get("rows"), list):
        raise ValueError("rows must be list")
    if int(payload.get("executed_seed_count", 0)) != len(payload.get("rows", [])):
        raise ValueError("executed_seed_count must match len(rows)")
    if not isinstance(payload.get("skipped"), bool):
        raise ValueError("skipped must be bool")

    dist = payload.get("distribution", {})
    if not isinstance(dist, dict):
        raise ValueError("distribution must be object")
    for key in (
        "raw_candidate_count_median",
        "activation_rate_median",
        "trade_count_median",
        "live_exp_lcb_median",
        "live_exp_lcb_worst",
        "live_exp_lcb_best",
    ):
        if key not in dist:
            raise ValueError(f"distribution.{key} missing")

    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            raise ValueError("row entries must be objects")
        for key in ("seed", "run_id", "raw_signal_count", "activation_rate", "trade_count", "live_best_exp_lcb"):
            if key not in row:
                raise ValueError(f"row.{key} missing")


def render_stage43_performance_report(payload: dict[str, Any]) -> str:
    """Render Stage-43 performance report."""

    baseline = dict(payload.get("baseline", {}))
    upgraded = dict(payload.get("upgraded", {}))
    delta = dict(payload.get("delta", {}))
    phase = dict(payload.get("phase_runtime_seconds", {}))

    lines = [
        "# Stage-43 Performance Report",
        "",
        "## Baseline vs Upgraded",
        f"- baseline_run_id: `{baseline.get('run_id', '')}`",
        f"- upgraded_run_id: `{upgraded.get('run_id', '')}`",
        f"- budget_mode: `{payload.get('budget_mode', '')}`",
        "",
        "| metric | baseline | upgraded | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in (
        "raw_signal_count",
        "activation_rate",
        "trade_count",
        "research_best_exp_lcb",
        "live_best_exp_lcb",
        "wf_executed_pct",
        "mc_trigger_pct",
        "runtime_seconds",
    ):
        lines.append(
            "| {k} | {b:.6f} | {u:.6f} | {d:.6f} |".format(
                k=key,
                b=float(baseline.get(key, 0.0)),
                u=float(upgraded.get(key, 0.0)),
                d=float(delta.get(f"delta_{key}", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Runtime By Phase (seconds)",
            "| phase | seconds |",
            "| --- | ---: |",
        ]
    )
    for key in REQUIRED_PHASE_KEYS:
        lines.append(f"| {key} | {float(phase.get(key, 0.0)):.6f} |")
    lines.extend(
        [
            "",
            f"- slowest_phase: `{payload.get('slowest_phase', '')}`",
            f"- promising: `{bool(payload.get('promising', False))}`",
            f"- note: `{payload.get('runtime_notes', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_stage43_five_seed_report(payload: dict[str, Any]) -> str:
    """Render Stage-43 five-seed report."""

    dist = dict(payload.get("distribution", {}))
    lines = [
        "# Stage-43 5-Seed Validation Report",
        "",
        f"- upgraded_reference_run_id: `{payload.get('upgraded_reference_run_id', '')}`",
        f"- skipped: `{bool(payload.get('skipped', False))}`",
        f"- skip_reason: `{payload.get('skip_reason', '')}`",
        f"- executed_seed_count: `{int(payload.get('executed_seed_count', 0))}`",
        "",
        "## Seed Rows",
        "| seed | run_id | raw_signal_count | activation_rate | trade_count | live_best_exp_lcb |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {seed} | {run_id} | {raw} | {act:.6f} | {trade:.6f} | {live:.6f} |".format(
                seed=int(row.get("seed", 0)),
                run_id=str(row.get("run_id", "")),
                raw=int(row.get("raw_signal_count", 0)),
                act=float(row.get("activation_rate", 0.0)),
                trade=float(row.get("trade_count", 0.0)),
                live=float(row.get("live_best_exp_lcb", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Distribution",
            f"- raw_candidate_count_median: `{float(dist.get('raw_candidate_count_median', 0.0)):.6f}`",
            f"- activation_rate_median: `{float(dist.get('activation_rate_median', 0.0)):.6f}`",
            f"- trade_count_median: `{float(dist.get('trade_count_median', 0.0)):.6f}`",
            f"- live_exp_lcb_median: `{float(dist.get('live_exp_lcb_median', 0.0)):.6f}`",
            f"- live_exp_lcb_worst: `{float(dist.get('live_exp_lcb_worst', 0.0)):.6f}`",
            f"- live_exp_lcb_best: `{float(dist.get('live_exp_lcb_best', 0.0)):.6f}`",
            "",
            "## Note",
            f"- {payload.get('note', '')}",
        ]
    )
    return "\n".join(lines).strip() + "\n"

