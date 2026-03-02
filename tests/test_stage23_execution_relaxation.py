from __future__ import annotations

import pandas as pd

from buffmini.stage23.order_builder import build_adaptive_orders


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 12,
            "high": [101.0] * 12,
            "low": [99.0] * 12,
            "close": [100.0] * 12,
            "volume": [1000.0, 10.0] + [1000.0] * 10,
            "atr_14": [1.0] * 12,
        }
    )


def _cfg() -> dict:
    return {
        "costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005},
        "cost_model": {"mode": "simple", "round_trip_cost_pct": 0.1},
        "risk": {"max_gross_exposure": 10.0},
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


def test_margin_fail_reduces_size_then_accepts() -> None:
    frame = _frame()
    signal = pd.Series([1] + [0] * 11, index=frame.index, dtype=int)
    score = pd.Series([50.0] + [0.0] * 11, index=frame.index, dtype=float)
    result = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=_cfg(), symbol="BTC/USDT")
    assert int((result["accepted_signal"] != 0).sum()) == 1
    events = [str(item.get("event", "")) for item in result["adjustment_events"]]
    assert "size_reduction_margin" in events


def test_hard_slippage_rejects_with_reason() -> None:
    frame = _frame()
    signal = pd.Series([1] + [0] * 11, index=frame.index, dtype=int)
    score = pd.Series([1.0] + [0.0] * 11, index=frame.index, dtype=float)
    cfg = _cfg()
    cfg["cost_model"] = {
        "mode": "v2",
        "round_trip_cost_pct": 0.1,
        "v2": {
            "slippage_bps_base": 120.0,
            "slippage_bps_vol_mult": 0.0,
            "spread_bps": 0.5,
            "delay_bars": 0,
            "vol_proxy": "atr_pct",
            "vol_lookback": 14,
            "max_total_bps_per_side": 150.0,
        },
    }
    result = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="BTC/USDT")
    assert int((result["accepted_signal"] != 0).sum()) == 0
    reasons = [str(item.get("reason", "")) for item in result["reject_events"]]
    assert "SLIPPAGE_TOO_HIGH" in reasons


def test_partial_fill_accepts_with_min_ratio() -> None:
    frame = _frame()
    signal = pd.Series([0, 1] + [0] * 10, index=frame.index, dtype=int)
    score = pd.Series([0.0, 1.0] + [0.0] * 10, index=frame.index, dtype=float)
    result = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=_cfg(), symbol="BTC/USDT")
    assert int((result["accepted_signal"] != 0).sum()) == 1
    fill_ratio = float(result["orders_df"]["fill_ratio"].iloc[0])
    assert 0.3 <= fill_ratio <= 1.0


def test_execution_relaxation_deterministic() -> None:
    frame = _frame()
    signal = pd.Series([1, 0, -1] + [0] * 9, index=frame.index, dtype=int)
    score = pd.Series([2.0, 0.0, 2.0] + [0.0] * 9, index=frame.index, dtype=float)
    cfg = _cfg()
    left = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="ETH/USDT")
    right = build_adaptive_orders(frame=frame, raw_side=signal, score=score, cfg=cfg, symbol="ETH/USDT")
    assert left["accepted_signal"].equals(right["accepted_signal"])
    assert left["orders_df"].equals(right["orders_df"])
    assert left["reject_events"] == right["reject_events"]
    assert left["adjustment_events"] == right["adjustment_events"]

