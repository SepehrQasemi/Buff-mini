from __future__ import annotations

from buffmini.stage52 import build_setup_candidate_v2, evaluate_family_coverage, validate_setup_candidate_v2


def test_stage52_builds_valid_candidate_v2() -> None:
    candidate = build_setup_candidate_v2(
        {
            "candidate_id": "src1",
            "family": "structure_pullback_continuation",
            "context": "trend",
            "trigger": "pullback_to_structure_level",
            "confirmation": "flow_continuation",
            "invalidation": "structure_break",
            "beam_score": 0.72,
            "modules": ["structure_engine"],
        },
        timeframe="1h",
        round_trip_cost_pct=0.001,
    )
    validate_setup_candidate_v2(candidate)
    assert candidate["eligible_for_replay"] is True
    assert float(candidate["rr_model"]["first_target_rr"]) >= 1.5
    assert str(candidate["economic_fingerprint"]).strip() != ""
    assert str(candidate["subfamily"]).strip() != ""
    assert str(candidate["risk_model"]).strip() != ""
    assert str(candidate["exit_family"]).strip() != ""
    assert int(candidate["time_stop_bars"]) >= 1


def test_stage52_rejects_missing_confirmation() -> None:
    candidate = build_setup_candidate_v2(
        {
            "candidate_id": "src2",
            "family": "liquidity_sweep_reversal",
            "context": "range",
            "trigger": "liquidity_sweep",
            "confirmation": "",
            "invalidation": "failed_reclaim",
            "beam_score": 0.65,
        },
        timeframe="30m",
        round_trip_cost_pct=0.001,
    )
    assert candidate["eligible_for_replay"] is False
    assert candidate["pre_replay_reject_reason"] == "REJECT::NO_CONFIRMATION"


def test_stage52_family_coverage_detects_missing_family() -> None:
    coverage = evaluate_family_coverage(
        {
            "family_counts": {
                "liquidity_sweep_reversal": 10,
                "squeeze_flow_breakout": 5,
            }
        },
        active_families=[
            "structure_pullback_continuation",
            "liquidity_sweep_reversal",
            "squeeze_flow_breakout",
        ],
    )
    assert coverage["family_coverage_ok"] is False
    assert coverage["missing_families"] == ["structure_pullback_continuation"]
