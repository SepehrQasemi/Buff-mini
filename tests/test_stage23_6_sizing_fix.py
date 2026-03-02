from __future__ import annotations

import pandas as pd

from buffmini.stage23.order_builder import build_adaptive_orders, round_qty_to_step


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
        "risk": {"max_gross_exposure": 5.0},
        "evaluation": {
            "stage10": {"evaluation": {"take_profit_atr_multiple": 3.0}},
            "stage23": {
                "enabled": True,
                "sizing_fix_enabled": True,
                "order_builder": {
                    "min_stop_atr_mult": 0.8,
                    "min_stop_bps": 8.0,
                    "min_rr": 0.8,
                    "min_trade_notional": 0.01,
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
                    "max_size_reduction_steps": 5,
                    "slippage_soft_threshold_bps": 15.0,
                    "slippage_hard_threshold_bps": 40.0,
                },
            },
        },
    }


def test_round_qty_to_step_contract() -> None:
    assert round_qty_to_step(0.0009, 0.001, "floor") == 0.0
    assert round_qty_to_step(0.0009, 0.001, "ceil") == 0.001
    assert round_qty_to_step(0.0009, 0.001, "nearest") == 0.001


def test_floor_to_zero_is_rescued_by_single_step_ceil() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.05 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    assert int((out["accepted_signal"] != 0).sum()) == 1
    trace = out["sizing_trace"]
    assert bool(trace["ceil_rescue_applied"].iloc[0]) is True
    assert float(trace["rounded_size_after"].iloc[0]) >= 0.001


def test_ceil_rescue_cap_violation_rejects_explicitly() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.05 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["risk"]["max_gross_exposure"] = 0.0005
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    reasons = [str(item.get("reason", "")) for item in out["reject_events"]]
    assert "POLICY_CAP_HIT" in reasons or "MARGIN_FAIL" in reasons
    assert "SIZE_ZERO" not in reasons


def test_min_notional_bump_works_and_respects_minimum() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["evaluation"]["stage23"]["order_builder"]["min_trade_notional"] = 10.0
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    assert int((out["accepted_signal"] != 0).sum()) == 1
    assert float(out["orders_df"]["final_notional"].iloc[0]) >= 10.0


def test_min_notional_bump_not_possible_returns_size_too_small() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["evaluation"]["stage23"]["order_builder"]["min_trade_notional"] = 10.0
    cfg["risk"]["max_gross_exposure"] = 0.05
    out = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    reasons = [str(item.get("reason", "")) for item in out["reject_events"]]
    assert "SIZE_TOO_SMALL" in reasons


def test_legacy_mode_can_emit_size_zero_but_fix_mode_cannot_for_same_case() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.05 / 100.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg_old = _cfg()
    cfg_old["evaluation"]["stage23"]["sizing_fix_enabled"] = False
    cfg_old["evaluation"]["stage23"]["sizing"]["allow_single_step_ceil_rescue"] = False
    old = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg_old, symbol="BTC/USDT")
    old_reasons = [str(item.get("reason", "")) for item in old["reject_events"]]
    assert "SIZE_ZERO" in old_reasons

    cfg_new = _cfg()
    new = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg_new, symbol="BTC/USDT")
    new_reasons = [str(item.get("reason", "")) for item in new["reject_events"]]
    assert "SIZE_ZERO" not in new_reasons
