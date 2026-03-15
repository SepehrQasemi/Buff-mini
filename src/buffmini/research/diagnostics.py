"""Candidate diagnostics shared across truth-lab, ranking, and campaign stages."""

from __future__ import annotations

import json
from typing import Any


def compute_candidate_risk_card(candidate: dict[str, Any], *, behavior_profile: dict[str, Any] | None = None) -> dict[str, float]:
    """Compute a deterministic pre-edge risk card from candidate geometry and behavior."""

    behavior = dict(behavior_profile or {})
    hold_bars = int(max(1, float(candidate.get("expected_hold_bars", candidate.get("time_stop_bars", 12)) or 12)))
    cost_edge = float(candidate.get("cost_edge_proxy", 0.0) or 0.0)
    rr_first_target = float(candidate.get("rr_first_target", 0.0) or 0.0)
    duplication_score = float(candidate.get("duplication_score", 0.0) or 0.0)
    activation_density = float(behavior.get("activation_density", candidate.get("activation_density", 0.0)) or 0.0)
    transfer_prior = float(candidate.get("transfer_risk_prior", 0.0) or 0.0)
    entry_overlap = float(behavior.get("entry_overlap_score", candidate.get("entry_overlap_score", 0.0)) or 0.0)
    exit_overlap = float(behavior.get("exit_overlap_score", candidate.get("exit_overlap_score", 0.0)) or 0.0)
    clustering_risk = float(behavior.get("clustering_risk", candidate.get("clustering_risk", 0.0)) or 0.0)
    failure_similarity = float(behavior.get("failure_pattern_similarity", candidate.get("failure_pattern_similarity", 0.0)) or 0.0)
    active_count = int(float(behavior.get("active_count", candidate.get("active_count", 0)) or 0))
    trade_density_expectation = str(candidate.get("trade_density_expectation", "")).strip().lower()
    transfer_expectation = str(candidate.get("transfer_expectation", "")).strip().lower()
    expected_failures = {str(item).strip().lower() for item in list(candidate.get("expected_failure_modes", []) or [])}
    exit_family = str(candidate.get("exit_family", candidate.get("target_logic", ""))).strip().lower()
    regime_map = _safe_json_mapping(behavior.get("regime_activation_map", candidate.get("regime_activation_map", "{}")))
    regime_concentration = float(max(regime_map.values()) if regime_map else candidate.get("regime_concentration_risk", 0.35) or 0.35)
    thin_evidence_base = max(0.0, 1.0 - min(1.0, active_count / 24.0))
    if trade_density_expectation == "low":
        thin_evidence_base = max(thin_evidence_base, 0.42)
    if "opportunity_sparsity" in expected_failures:
        thin_evidence_base = min(1.0, thin_evidence_base + 0.12)
    thin_evidence_risk = float(candidate.get("thin_evidence_risk", thin_evidence_base) or 0.0)
    trade_density_floor = 0.0 if trade_density_expectation == "high" else 0.10 if trade_density_expectation == "medium" else 0.28
    cost_fragility_floor = 0.0
    if "cost_fragility" in expected_failures or "cost_decay" in expected_failures:
        cost_fragility_floor = 0.10
    hold_sanity = float(min(1.0, abs(hold_bars - 16) / 32.0))
    if exit_family == "time_exit" and hold_bars >= 16:
        hold_sanity = min(1.0, hold_sanity + 0.08)
    regime_floor = 0.18 if len([value for value in regime_map.values() if value > 0.10]) <= 1 else 0.0
    transfer_floor = 0.0 if transfer_expectation == "high" else 0.08 if transfer_expectation == "moderate" else 0.18
    risk = {
        "trade_density_risk": float(max(trade_density_floor, max(0.0, 1.0 - min(1.0, activation_density * 8.0)))),
        "cost_fragility_risk": float(max(cost_fragility_floor, max(0.0, min(1.0, 0.5 - (cost_edge * 40.0))))),
        "regime_concentration_risk": float(max(regime_floor, max(0.0, min(1.0, regime_concentration)))),
        "hold_sanity_risk": hold_sanity,
        "overlap_duplication_risk": float(max(0.0, min(1.0, max(duplication_score, entry_overlap, exit_overlap)))),
        "clustering_risk": float(max(0.0, min(1.0, max(clustering_risk, failure_similarity)))),
        "thin_evidence_risk": float(max(0.0, min(1.0, thin_evidence_risk))),
        "transfer_risk_prior": float(max(transfer_floor, max(0.0, min(1.0, transfer_prior)))),
        "rr_adequacy_bonus": float(max(0.0, min(1.0, rr_first_target / 3.0))),
    }
    risk["aggregate_risk"] = float(
        (
            risk["trade_density_risk"]
            + risk["cost_fragility_risk"]
            + risk["regime_concentration_risk"]
            + risk["hold_sanity_risk"]
            + risk["overlap_duplication_risk"]
            + risk["clustering_risk"]
            + risk["thin_evidence_risk"]
            + risk["transfer_risk_prior"]
        )
        / 8.0
    )
    return risk


def classify_candidate_hierarchy(
    *,
    rank_score: float,
    replay_exp_lcb: float,
    walkforward_usable_windows: int,
    decision_use_allowed: bool,
    aggregate_risk: float,
    robustness_level: int = 0,
) -> str:
    """Assign a richer candidate hierarchy without overstating evidence quality."""

    if int(max(0, robustness_level)) >= 3:
        return "robust_candidate"
    if decision_use_allowed and replay_exp_lcb > 0.0 and walkforward_usable_windows >= 3 and aggregate_risk <= 0.45:
        return "validated_candidate"
    if rank_score >= 0.45 and replay_exp_lcb >= -0.005 and aggregate_risk <= 0.60:
        return "promising_but_unproven"
    if rank_score >= 0.22 and replay_exp_lcb >= -0.015 and aggregate_risk <= 0.85:
        return "interesting_but_fragile"
    return "junk"


def classify_candidate_tier(
    *,
    rank_score: float,
    replay_exp_lcb: float,
    walkforward_usable_windows: int,
    decision_use_allowed: bool,
    aggregate_risk: float,
) -> str:
    """Assign a bounded candidate class without pretending validation is complete."""

    hierarchy = classify_candidate_hierarchy(
        rank_score=rank_score,
        replay_exp_lcb=replay_exp_lcb,
        walkforward_usable_windows=walkforward_usable_windows,
        decision_use_allowed=decision_use_allowed,
        aggregate_risk=aggregate_risk,
        robustness_level=0,
    )
    if hierarchy in {"robust_candidate", "validated_candidate"}:
        return "validated_candidate"
    if hierarchy == "promising_but_unproven":
        return "promising_but_unproven"
    return "rejected"


def compute_near_miss_distance(
    *,
    replay_trade_count: int,
    replay_exp_lcb: float,
    walkforward_usable_windows: int,
    robustness_level: int,
    transfer_classification: str,
    config: dict[str, Any],
) -> float:
    """Distance-to-survival score; lower means closer to passing."""

    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    walkforward_gate = dict(config.get("promotion_gates", {}).get("walkforward", {}))
    min_trades = max(1, int(replay_gate.get("min_trade_count", 40)))
    min_exp = float(replay_gate.get("min_exp_lcb", 0.0))
    min_windows = max(1, int(walkforward_gate.get("min_usable_windows", 3)))
    trade_gap = max(0.0, (float(min_trades) - float(max(0, replay_trade_count))) / float(min_trades))
    exp_gap = max(0.0, min_exp - float(replay_exp_lcb))
    wf_gap = max(0.0, float(min_windows - max(0, walkforward_usable_windows)) / float(min_windows))
    robust_gap = max(0.0, float(3 - max(0, robustness_level)) / 3.0)
    neutral_transfer = {"transferable", "partially_transferable", "not_evaluated", "transfer_skipped", "not_required"}
    transfer_gap = 0.0 if str(transfer_classification) in neutral_transfer else 0.20
    return float(round(trade_gap + exp_gap + wf_gap + robust_gap + transfer_gap, 6))


def resolve_first_death_stage(
    *,
    replay_trade_count: int,
    replay_exp_lcb: float,
    replay_allowed: bool,
    walkforward_usable_windows: int,
    monte_carlo_passed: bool,
    perturbation_passed: bool,
    split_passed: bool,
    transfer_classification: str,
    market_blocked_reason: str = "",
    config: dict[str, Any],
) -> tuple[str, str]:
    """Resolve the first stage where a candidate effectively dies."""

    if str(market_blocked_reason).strip():
        return "data_gate", str(market_blocked_reason).strip()
    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    walkforward_gate = dict(config.get("promotion_gates", {}).get("walkforward", {}))
    if (not replay_allowed) or replay_trade_count < int(replay_gate.get("min_trade_count", 40)) or replay_exp_lcb < float(replay_gate.get("min_exp_lcb", 0.0)):
        if not replay_allowed:
            return "replay", "replay_blocked"
        if replay_trade_count < int(replay_gate.get("min_trade_count", 40)):
            return "replay", "min_trade_count"
        return "replay", "exp_lcb"
    if walkforward_usable_windows < int(walkforward_gate.get("min_usable_windows", 3)):
        return "walkforward", "usable_windows"
    if not monte_carlo_passed:
        return "monte_carlo", "stress_fail"
    if not perturbation_passed:
        return "cross_perturbation", "perturbation_fail"
    if not split_passed:
        return "split_perturbation", "split_fail"
    if str(transfer_classification) not in {"transferable", "partially_transferable", "not_evaluated", "transfer_skipped", "not_required"}:
        return "transfer", str(transfer_classification)
    return "survived", ""


def _safe_json_mapping(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        out: dict[str, float] = {}
        for key, value in raw.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        return out
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out
