from __future__ import annotations

import pandas as pd

from buffmini.stage23.order_builder import build_adaptive_orders


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.0] * 6,
            "volume": [1000.0] * 6,
            "atr_14": [1.0] * 6,
        }
    )


def _cfg() -> dict:
    return {
        "costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005},
        "cost_model": {"mode": "simple", "round_trip_cost_pct": 0.1},
        "risk": {"max_gross_exposure": 0.01},
        "evaluation": {
            "stage10": {"evaluation": {"take_profit_atr_multiple": 3.0}},
            "stage23": {
                "enabled": True,
                "sizing_fix_enabled": True,
                "order_builder": {
                    "min_stop_atr_mult": 0.8,
                    "min_stop_bps": 8.0,
                    "min_rr": 0.8,
                    "min_trade_notional": 10.0,
                    "min_trade_qty": 0.0,
                    "qty_step": 0.001,
                    "allow_size_bump_to_min_notional": True,
                    "rr_fallback_exit_mode": "fixed_atr",
                },
                "sizing": {
                    "qty_rounding_default": "floor",
                    "qty_rounding_on_min_notional_bump": "ceil",
                    "allow_single_step_ceil_rescue": True,
                    "ceil_rescue_max_overage_steps": 1,
                },
                "execution": {
                    "allow_partial_fill": True,
                    "partial_fill_min_ratio": 0.3,
                    "allow_size_reduction_on_margin_fail": True,
                    "max_size_reduction_steps": 0,
                    "slippage_soft_threshold_bps": 15.0,
                    "slippage_hard_threshold_bps": 40.0,
                },
            },
        },
    }


def test_policy_cap_binding_rejects_explicitly_not_zero() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["evaluation"]["stage23"]["execution"]["allow_size_reduction_on_margin_fail"] = False
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    reasons = [str(item.get("reason", "")) for item in out["reject_events"]]
    assert "POLICY_CAP_HIT" in reasons
    assert "SIZE_ZERO" not in reasons


def test_margin_binding_rejects_explicitly_not_zero() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["evaluation"]["stage23"]["execution"]["allow_size_reduction_on_margin_fail"] = True
    cfg["evaluation"]["stage23"]["execution"]["max_size_reduction_steps"] = 0
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    reasons = [str(item.get("reason", "")) for item in out["reject_events"]]
    assert "MARGIN_FAIL" in reasons
    assert "SIZE_ZERO" not in reasons


def test_breakdown_totals_stay_consistent() -> None:
    frame = _frame()
    signal = pd.Series([1, -1, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([3.0, 3.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=_cfg(), symbol="ETH/USDT")
    payload = dict(out["breakdown"])
    attempted = int(payload["total_orders_attempted"])
    accepted = int(payload["total_orders_accepted"])
    rejected = int(payload["total_orders_rejected"])
    assert attempted == accepted + rejected
