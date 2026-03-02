from __future__ import annotations

import pandas as pd

from buffmini.stage23.order_builder import build_adaptive_orders


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "volume": [1000.0] * 8,
            "atr_14": [0.01] * 8,
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
                "order_builder": {
                    "min_stop_atr_mult": 0.8,
                    "min_stop_bps": 8.0,
                    "min_rr": 0.8,
                    "min_trade_notional": 10.0,
                    "allow_size_bump_to_min_notional": True,
                    "rr_fallback_exit_mode": "fixed_atr",
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


def test_tight_stop_is_clamped_not_rejected() -> None:
    frame = _frame()
    signal = pd.Series([0, 1, 0, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    result = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=_cfg(), symbol="BTC/USDT")
    assert int((result["accepted_signal"] != 0).sum()) == 1
    assert not result["orders_df"].empty
    assert float(result["orders_df"]["stop_distance_bps"].iloc[0]) >= 8.0


def test_rr_invalid_fallback_or_reject() -> None:
    frame = _frame()
    signal = pd.Series([1] + [0] * 7, index=frame.index, dtype=int)
    score = pd.Series([1.0] + [0.0] * 7, index=frame.index, dtype=float)

    cfg_ok = _cfg()
    cfg_ok["evaluation"]["stage23"]["order_builder"]["min_rr"] = 10.0
    cfg_ok["evaluation"]["stage23"]["order_builder"]["rr_fallback_exit_mode"] = "fixed_atr"
    accepted = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg_ok, symbol="BTC/USDT")
    assert int((accepted["accepted_signal"] != 0).sum()) == 1

    cfg_bad = _cfg()
    cfg_bad["evaluation"]["stage23"]["order_builder"]["min_rr"] = 10.0
    cfg_bad["evaluation"]["stage23"]["order_builder"]["rr_fallback_exit_mode"] = ""
    rejected = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg_bad, symbol="BTC/USDT")
    reasons = [str(item.get("reason", "")) for item in rejected["reject_events"]]
    assert "RR_INVALID" in reasons


def test_min_size_bump_and_cap_reject() -> None:
    frame = _frame()
    signal = pd.Series([1] + [0] * 7, index=frame.index, dtype=int)
    score = pd.Series([0.01] + [0.0] * 7, index=frame.index, dtype=float)

    cfg_bump = _cfg()
    bumped = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg_bump, symbol="BTC/USDT")
    assert int((bumped["accepted_signal"] != 0).sum()) == 1
    assert float(bumped["orders_df"]["filled_notional"].iloc[0]) >= 10.0

    cfg_cap = _cfg()
    cfg_cap["risk"]["max_gross_exposure"] = 0.01
    capped = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg_cap, symbol="BTC/USDT")
    assert int((capped["accepted_signal"] != 0).sum()) == 0
    reasons = [str(item.get("reason", "")) for item in capped["reject_events"]]
    assert any(reason in {"MARGIN_FAIL", "POLICY_CAP_HIT", "SIZE_TOO_SMALL"} for reason in reasons)

