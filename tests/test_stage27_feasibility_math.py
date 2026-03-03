from __future__ import annotations

import math

from buffmini.execution.feasibility import explain_reject, min_required_equity, min_required_risk_pct


def test_min_required_risk_pct_monotonic_with_equity_and_cost() -> None:
    low_eq = min_required_risk_pct(
        equity=1000.0,
        min_notional=10.0,
        stop_dist_pct=0.01,
        cost_rt_pct=0.001,
        max_notional_pct=1.0,
    )
    high_eq = min_required_risk_pct(
        equity=10000.0,
        min_notional=10.0,
        stop_dist_pct=0.01,
        cost_rt_pct=0.001,
        max_notional_pct=1.0,
    )
    higher_cost = min_required_risk_pct(
        equity=1000.0,
        min_notional=10.0,
        stop_dist_pct=0.01,
        cost_rt_pct=0.005,
        max_notional_pct=1.0,
    )
    assert high_eq < low_eq
    assert higher_cost > low_eq


def test_min_required_equity_inverse_sanity() -> None:
    equity = min_required_equity(
        risk_pct=0.05,
        min_notional=10.0,
        stop_dist_pct=0.02,
        cost_rt_pct=0.001,
        max_notional_pct=1.0,
    )
    required_risk = min_required_risk_pct(
        equity=equity,
        min_notional=10.0,
        stop_dist_pct=0.02,
        cost_rt_pct=0.001,
        max_notional_pct=1.0,
    )
    assert required_risk <= 0.05 + 1e-9


def test_explain_reject_payload_keys() -> None:
    payload = explain_reject(
        {
            "equity": 1000.0,
            "risk_pct_used": 0.02,
            "min_notional": 10.0,
            "stop_dist_pct": 0.01,
            "cost_rt_pct": 0.001,
            "max_notional_pct": 1.0,
        }
    )
    assert "minimum_required_risk_pct" in payload
    assert "minimum_required_equity" in payload
    assert math.isfinite(float(payload["minimum_required_risk_pct"]))

