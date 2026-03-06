"""Stage-37 reporting schema helpers."""

from __future__ import annotations

from typing import Any


def validate_stage37_engine_summary(payload: dict[str, Any]) -> None:
    """Validate Stage-37 engine summary schema contract."""

    required_top = {"stage", "seed", "baseline", "upgraded", "delta", "promising"}
    missing = required_top.difference(payload.keys())
    if missing:
        raise ValueError(f"Missing Stage-37 engine summary keys: {sorted(missing)}")
    if str(payload.get("stage")) != "37.4":
        raise ValueError("stage must be '37.4'")
    for key in ("baseline", "upgraded"):
        section = payload.get(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"{key} must be an object")
        for metric in ("run_id", "wf_executed_pct", "mc_trigger_pct", "raw_signal_count", "activation_rate", "trade_count", "research_best_exp_lcb", "live_best_exp_lcb", "maxDD"):
            if metric not in section:
                raise ValueError(f"{key}.{metric} missing")
    delta = payload.get("delta", {})
    if not isinstance(delta, dict):
        raise ValueError("delta must be an object")
    for metric in (
        "delta_wf_executed_pct",
        "delta_mc_trigger_pct",
        "delta_raw_signal_count",
        "delta_activation_rate",
        "delta_trade_count",
        "delta_research_best_exp_lcb",
        "delta_live_best_exp_lcb",
        "delta_maxDD",
    ):
        if metric not in delta:
            raise ValueError(f"delta.{metric} missing")
    if not isinstance(payload.get("promising"), bool):
        raise ValueError("promising must be bool")
