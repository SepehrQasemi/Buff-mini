from __future__ import annotations

import pandas as pd

from buffmini.stage23.order_builder import build_adaptive_orders


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC"),
            "open": [100.0, 100.2, 100.1, 100.4, 100.3, 100.5],
            "high": [100.8, 100.9, 100.7, 101.0, 100.9, 101.1],
            "low": [99.3, 99.6, 99.5, 99.8, 99.7, 99.9],
            "close": [100.0, 100.2, 100.1, 100.4, 100.3, 100.5],
            "volume": [3000.0, 3100.0, 3050.0, 3200.0, 3150.0, 3300.0],
            "atr_14": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def _cfg(max_risk_cap: float) -> dict:
    return {
        "costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005},
        "cost_model": {"mode": "simple", "round_trip_cost_pct": 0.1},
        "risk": {"max_gross_exposure": 5.0},
        "execution": {
            "risk_auto_bump": {"enabled": True, "max_risk_cap": float(max_risk_cap)},
        },
        "evaluation": {
            "stage10": {"evaluation": {"take_profit_atr_multiple": 3.0}},
            "stage23": {
                "enabled": True,
                "order_builder": {
                    "min_trade_notional": 10.0,
                    "allow_size_bump_to_min_notional": False,
                },
            },
            "stage24": {
                "enabled": True,
                "sizing": {
                    "mode": "risk_pct",
                    "risk_pct_user": 0.0001,
                    "risk_ladder": {
                        "enabled": True,
                        "r_min": 0.000001,
                        "r_max": 0.5,
                        "e_ref": 1000.0,
                        "r_ref": 0.08,
                        "k": 0.5,
                    },
                },
                "order_constraints": {
                    "min_trade_notional": 10.0,
                    "allow_size_bump_to_min_notional": False,
                    "max_notional_pct_of_equity": 1.0,
                },
                "simulation": {"initial_equities": [100.0]},
            },
            "constraints": {
                "mode": "live",
                "live": {"min_trade_notional": 10.0, "min_trade_qty": 0.0, "qty_step": 0.001},
                "research": {"min_trade_notional": 0.0, "min_trade_qty": 0.0, "qty_step": 0.0},
            },
            "modes": {
                "research": {
                    "min_notional_override": 1.0,
                    "ignore_exchange_precision": True,
                    "enforce_margin_caps": False,
                    "enforce_min_notional": False,
                    "enforce_size_step": False,
                },
                "live": {"use_exchange_rules": True},
            },
        },
    }


def test_stage27_9_risk_bump_accepts_when_floor_under_cap() -> None:
    frame = _frame()
    raw_side = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    out = build_adaptive_orders(frame=frame, raw_side=raw_side, score=score, cfg=_cfg(0.20), symbol="BTC/USDT")
    assert int((out["accepted_signal"] != 0).sum()) == 1
    events = list(out.get("risk_bump_events", []))
    assert events
    assert str(events[0].get("status_after_bump", "")) == "VALID"


def test_stage27_9_risk_bump_rejects_when_cap_too_low() -> None:
    frame = _frame()
    raw_side = pd.Series([1, 0, 0, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    out = build_adaptive_orders(frame=frame, raw_side=raw_side, score=score, cfg=_cfg(0.0002), symbol="BTC/USDT")
    reasons = [str(item.get("reason", "")) for item in out.get("reject_events", [])]
    assert "EXECUTION_INFEASIBLE_CAP" in reasons
