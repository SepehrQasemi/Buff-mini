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
    regime_map = _safe_json_mapping(behavior.get("regime_activation_map", candidate.get("regime_activation_map", "{}")))
    regime_concentration = float(max(regime_map.values()) if regime_map else candidate.get("regime_concentration_risk", 0.35) or 0.35)
    thin_evidence_risk = float(candidate.get("thin_evidence_risk", max(0.0, 1.0 - min(1.0, active_count / 24.0))) or 0.0)
    risk = {
        "trade_density_risk": float(max(0.0, 1.0 - min(1.0, activation_density * 8.0))),
        "cost_fragility_risk": float(max(0.0, min(1.0, 0.5 - (cost_edge * 40.0)))),
        "regime_concentration_risk": float(max(0.0, min(1.0, regime_concentration))),
        "hold_sanity_risk": float(min(1.0, abs(hold_bars - 16) / 32.0)),
        "overlap_duplication_risk": float(max(0.0, min(1.0, max(duplication_score, entry_overlap, exit_overlap)))),
        "clustering_risk": float(max(0.0, min(1.0, max(clustering_risk, failure_similarity)))),
        "thin_evidence_risk": float(max(0.0, min(1.0, thin_evidence_risk))),
        "transfer_risk_prior": float(max(0.0, min(1.0, transfer_prior))),
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


def classify_candidate_tier(
    *,
    rank_score: float,
    replay_exp_lcb: float,
    walkforward_usable_windows: int,
    decision_use_allowed: bool,
    aggregate_risk: float,
) -> str:
    """Assign a bounded candidate class without pretending validation is complete."""

    if decision_use_allowed and replay_exp_lcb > 0.0 and walkforward_usable_windows >= 3 and aggregate_risk <= 0.45:
        return "validated_candidate"
    if rank_score >= 0.45 and replay_exp_lcb >= -0.005 and aggregate_risk <= 0.60:
        return "promising_but_unproven"
    if rank_score >= 0.25 and replay_exp_lcb >= 0.0 and aggregate_risk <= 0.70:
        return "promising_but_unproven"
    return "rejected"


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
