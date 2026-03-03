from __future__ import annotations

import pandas as pd

from buffmini.stage23.order_builder import build_adaptive_orders


def test_feasibility_fields_attached_to_size_rejects() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
            "atr_14": [1.0, 1.0, 1.0, 1.0],
        }
    )
    raw_side = pd.Series([1, 0, 0, 0], index=frame.index, dtype=int)
    score = pd.Series([0.0001, 0.0, 0.0, 0.0], index=frame.index, dtype=float)
    cfg = {
        "evaluation": {
            "stage23": {
                "enabled": True,
                "order_builder": {
                    "min_trade_notional": 10.0,
                    "allow_size_bump_to_min_notional": False,
                },
            },
            "constraints": {
                "mode": "live",
                "live": {
                    "min_trade_notional": 10.0,
                    "min_trade_qty": 0.0,
                    "qty_step": 0.001,
                },
                "research": {
                    "min_trade_notional": 0.0,
                    "min_trade_qty": 0.0,
                    "qty_step": 0.0,
                },
            },
            "stage24": {"enabled": False},
        },
        "risk": {"max_gross_exposure": 5.0},
        "costs": {"round_trip_cost_pct": 0.1, "slippage_pct": 0.0005},
    }
    out = build_adaptive_orders(frame=frame, raw_side=raw_side, score=score, cfg=cfg, symbol="BTC/USDT")
    reject_events = list(out.get("reject_events", []))
    assert reject_events
    size_rejects = [row for row in reject_events if str(row.get("reason", "")) in {"SIZE_TOO_SMALL", "POLICY_CAP_HIT", "MARGIN_FAIL"}]
    assert size_rejects
    sample = size_rejects[0]
    assert "minimum_required_risk_pct" in sample
    assert "minimum_required_equity" in sample

