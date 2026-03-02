from __future__ import annotations

from buffmini.execution.margin_model import (
    PolicyCaps,
    apply_exposure_caps,
    compute_margin_required,
    is_trade_feasible,
)


def test_equity_scaling_invariant_for_percent_sizing() -> None:
    caps = PolicyCaps(
        max_notional_pct_of_equity=10.0,
        max_gross_exposure_mult=10.0,
        absolute_max_notional=0.0,
        margin_alloc_limit=1.0,
    )
    leverage = 2.0
    fees = 0.001
    buffer = 0.01

    equity_small = 1_000.0
    equity_large = 10_000.0
    desired_small = equity_small * 0.4
    desired_large = equity_large * 0.4

    capped_small, reason_small, _ = apply_exposure_caps(
        desired_notional=desired_small,
        policy_caps=caps,
        current_exposure=0.0,
        equity=equity_small,
    )
    capped_large, reason_large, _ = apply_exposure_caps(
        desired_notional=desired_large,
        policy_caps=caps,
        current_exposure=0.0,
        equity=equity_large,
    )
    assert reason_small == ""
    assert reason_large == ""
    assert capped_large == capped_small * 10.0

    margin_small = compute_margin_required(capped_small, leverage, fees, buffer)
    margin_large = compute_margin_required(capped_large, leverage, fees, buffer)
    assert margin_large == margin_small * 10.0

    ok_small, r_small, _ = is_trade_feasible(
        equity=equity_small,
        capped_notional=capped_small,
        leverage=leverage,
        margin_required=margin_small,
        policy_caps=caps,
    )
    ok_large, r_large, _ = is_trade_feasible(
        equity=equity_large,
        capped_notional=capped_large,
        leverage=leverage,
        margin_required=margin_large,
        policy_caps=caps,
    )
    assert ok_small is True
    assert ok_large is True
    assert r_small == ""
    assert r_large == ""


def test_ordering_invariant_caps_before_margin_reason_is_policy() -> None:
    caps = PolicyCaps(
        max_notional_pct_of_equity=0.5,
        max_gross_exposure_mult=0.5,
        absolute_max_notional=0.0,
        margin_alloc_limit=1.0,
    )
    equity = 1_000.0
    capped, reason, details = apply_exposure_caps(
        desired_notional=2_000.0,
        policy_caps=caps,
        current_exposure=500.0,
        equity=equity,
    )
    assert reason == "POLICY_CAP_HIT"
    assert capped == 0.0
    assert details["available_notional"] == 0.0

    margin_required = compute_margin_required(capped, leverage=2.0, fees_estimate=0.001, buffer=0.01)
    feasible, fail_reason, _ = is_trade_feasible(
        equity=equity,
        capped_notional=capped,
        leverage=2.0,
        margin_required=margin_required,
        policy_caps=caps,
    )
    assert feasible is False
    assert fail_reason == "POLICY_CAP_HIT"


def test_margin_model_is_deterministic() -> None:
    caps = PolicyCaps(
        max_notional_pct_of_equity=1.0,
        max_gross_exposure_mult=1.0,
        absolute_max_notional=0.0,
        margin_alloc_limit=0.8,
    )
    out1 = apply_exposure_caps(900.0, caps, current_exposure=0.0, equity=1_000.0)
    out2 = apply_exposure_caps(900.0, caps, current_exposure=0.0, equity=1_000.0)
    assert out1 == out2
    m1 = compute_margin_required(900.0, leverage=3.0, fees_estimate=0.001, buffer=0.01)
    m2 = compute_margin_required(900.0, leverage=3.0, fees_estimate=0.001, buffer=0.01)
    assert m1 == m2
    f1 = is_trade_feasible(1_000.0, 900.0, 3.0, m1, caps)
    f2 = is_trade_feasible(1_000.0, 900.0, 3.0, m2, caps)
    assert f1 == f2
