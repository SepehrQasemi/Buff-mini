"""Stage-50 and master summary schema validators."""

from __future__ import annotations

from typing import Any


def validate_stage50_performance_summary(payload: dict[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "baseline_runtime_seconds",
        "upgraded_runtime_seconds",
        "delta_runtime_seconds",
        "slowest_phase",
        "baseline_raw_signals",
        "upgraded_raw_signals",
        "baseline_trade_count",
        "upgraded_trade_count",
        "research_best_exp_lcb_before",
        "research_best_exp_lcb_after",
        "live_best_exp_lcb_before",
        "live_best_exp_lcb_after",
        "promising",
        "summary_hash",
    }
    missing = sorted(required.difference(set(payload.keys())))
    if missing:
        raise ValueError(f"Missing Stage-50 performance keys: {missing}")
    if str(payload.get("stage", "")) != "50":
        raise ValueError("stage must be '50'")


def validate_stage50_5seed_summary(payload: dict[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "skipped",
        "skip_reason_if_any",
        "executed_seeds",
        "activation_rate_distribution",
        "trade_count_distribution",
        "exp_lcb_distribution",
        "family_consistency",
        "summary_hash",
    }
    missing = sorted(required.difference(set(payload.keys())))
    if missing:
        raise ValueError(f"Missing Stage-50 5seed keys: {missing}")
    if str(payload.get("stage", "")) != "50_5seed":
        raise ValueError("stage must be '50_5seed'")
    if not isinstance(payload.get("executed_seeds"), list):
        raise ValueError("executed_seeds must be list")


def validate_stage44_50_master_summary(payload: dict[str, Any]) -> None:
    required = {
        "stage44_status",
        "stage45_status",
        "stage46_status",
        "stage47_status",
        "stage48_status",
        "stage49_status",
        "stage50_status",
        "stage47_raw_candidates_before",
        "stage47_raw_candidates_after",
        "stage48_stage_a_survivors",
        "stage48_stage_b_survivors",
        "stage49_registry_rows",
        "stage49_elites_count",
        "stage50_runtime_seconds_baseline",
        "stage50_runtime_seconds_upgraded",
        "stage50_promising",
        "stage50_5seed_executed",
        "deterministic_summary_hash",
        "final_verdict",
        "biggest_remaining_bottleneck",
        "next_cheapest_action",
    }
    missing = sorted(required.difference(set(payload.keys())))
    if missing:
        raise ValueError(f"Missing Stage44-50 master keys: {missing}")

