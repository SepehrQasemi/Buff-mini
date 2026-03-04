from __future__ import annotations

from buffmini.execution.feasibility import explain_reject
from buffmini.execution.feasibility_floor import calculate_min_equity, calculate_min_risk_pct


def test_stage27_9_calculate_min_risk_pct_monotonic() -> None:
    base = calculate_min_risk_pct(
        equity=1000.0,
        stop_distance_pct=0.01,
        min_notional=10.0,
        fee_roundtrip_pct=0.001,
        size_step=0.0,
    )
    higher_notional = calculate_min_risk_pct(
        equity=1000.0,
        stop_distance_pct=0.01,
        min_notional=20.0,
        fee_roundtrip_pct=0.001,
        size_step=0.0,
    )
    higher_cost = calculate_min_risk_pct(
        equity=1000.0,
        stop_distance_pct=0.02,
        min_notional=10.0,
        fee_roundtrip_pct=0.001,
        size_step=0.0,
    )
    assert higher_notional > base > 0.0
    assert higher_cost > base


def test_stage27_9_calculate_min_equity_monotonic() -> None:
    eq_low = calculate_min_equity(
        risk_pct=0.01,
        stop_distance_pct=0.02,
        min_notional=10.0,
    )
    eq_high = calculate_min_equity(
        risk_pct=0.005,
        stop_distance_pct=0.02,
        min_notional=10.0,
    )
    assert eq_low > 0.0
    assert eq_high > eq_low


def test_stage27_9_explain_reject_contains_floor_alias_fields() -> None:
    payload = explain_reject(
        {
            "equity": 1000.0,
            "risk_pct_used": 0.01,
            "min_notional": 10.0,
            "stop_dist_pct": 0.01,
            "cost_rt_pct": 0.001,
            "max_notional_pct": 1.0,
        }
    )
    assert "min_risk_required" in payload
    assert "min_equity_required" in payload
    assert "stop_distance" in payload
    assert "fee_rt_pct" in payload
    assert payload["min_risk_required"] == payload["minimum_required_risk_pct"]
    assert payload["min_equity_required"] == payload["minimum_required_equity"]

