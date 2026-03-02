from __future__ import annotations

from buffmini.stage24.sizing import (
    compute_notional_alloc_pct,
    compute_notional_risk_pct,
    compute_risk_pct,
    is_known_reject_reason,
)


def _cfg() -> dict:
    return {
        "evaluation": {
            "stage24": {
                "sizing": {
                    "mode": "risk_pct",
                    "alloc_pct": 0.25,
                    "risk_pct_user": None,
                    "risk_ladder": {
                        "enabled": True,
                        "r_min": 0.02,
                        "r_max": 0.20,
                        "e_ref": 1000.0,
                        "r_ref": 0.08,
                        "k": 0.5,
                    },
                    "clamps": {
                        "dd_soft": 0.10,
                        "dd_hard": 0.20,
                        "dd_soft_mult": 0.7,
                        "dd_hard_mult": 0.4,
                        "losing_streak_soft": 3,
                        "losing_streak_hard": 5,
                        "streak_soft_mult": 0.7,
                        "streak_hard_mult": 0.4,
                    },
                },
                "order_constraints": {
                    "min_trade_notional": 10.0,
                    "allow_size_bump_to_min_notional": True,
                    "max_notional_pct_of_equity": 1.0,
                },
            }
        }
    }


def test_ladder_monotonicity_by_equity() -> None:
    cfg = _cfg()
    low, _ = compute_risk_pct(equity=100.0, dd=0.0, losing_streak=0, cfg=cfg)
    mid, _ = compute_risk_pct(equity=1000.0, dd=0.0, losing_streak=0, cfg=cfg)
    high, _ = compute_risk_pct(equity=10000.0, dd=0.0, losing_streak=0, cfg=cfg)
    assert low >= mid >= high


def test_drawdown_and_streak_clamps_apply() -> None:
    cfg = _cfg()
    base, _ = compute_risk_pct(equity=1000.0, dd=0.0, losing_streak=0, cfg=cfg)
    clamped, parts = compute_risk_pct(equity=1000.0, dd=0.25, losing_streak=6, cfg=cfg)
    assert clamped < base
    assert float(parts["dd_mult"]) <= 0.4
    assert float(parts["streak_mult"]) <= 0.4


def test_cost_aware_notional_decreases_with_cost() -> None:
    constraints = _cfg()["evaluation"]["stage24"]["order_constraints"]
    low_cost, status_low, _, _ = compute_notional_risk_pct(
        equity=1000.0,
        risk_pct_used=0.02,
        stop_distance_pct=0.02,
        cost_rt_pct=0.001,
        constraints_cfg=constraints,
    )
    high_cost, status_high, _, _ = compute_notional_risk_pct(
        equity=1000.0,
        risk_pct_used=0.02,
        stop_distance_pct=0.02,
        cost_rt_pct=0.005,
        constraints_cfg=constraints,
    )
    assert status_low == "VALID" and status_high == "VALID"
    assert low_cost > high_cost


def test_min_notional_bump_and_reject_paths() -> None:
    constraints = _cfg()["evaluation"]["stage24"]["order_constraints"]
    n_ok, status_ok, reason_ok, details_ok = compute_notional_alloc_pct(
        equity=100.0,
        alloc_pct=0.01,
        constraints_cfg=constraints,
    )
    assert status_ok == "VALID"
    assert reason_ok == ""
    assert n_ok >= 10.0
    assert bool(details_ok["bumped_to_min_notional"]) is True

    blocked = dict(constraints)
    blocked["max_notional_pct_of_equity"] = 0.05
    n_bad, status_bad, reason_bad, _ = compute_notional_alloc_pct(
        equity=100.0,
        alloc_pct=0.01,
        constraints_cfg=blocked,
    )
    assert status_bad == "INVALID"
    assert n_bad == 0.0
    assert reason_bad == "SIZE_TOO_SMALL"
    assert is_known_reject_reason(reason_bad)


def test_alloc_mode_contract() -> None:
    constraints = _cfg()["evaluation"]["stage24"]["order_constraints"]
    notional, status, reason, details = compute_notional_alloc_pct(
        equity=1000.0,
        alloc_pct=0.25,
        constraints_cfg=constraints,
    )
    assert status == "VALID"
    assert reason == ""
    assert abs(notional - 250.0) < 1e-9
    assert float(details["notional_raw"]) == 250.0


def test_sizing_deterministic() -> None:
    cfg = _cfg()
    left = compute_risk_pct(equity=777.0, dd=0.13, losing_streak=4, cfg=cfg)
    right = compute_risk_pct(equity=777.0, dd=0.13, losing_streak=4, cfg=cfg)
    assert left == right
