from __future__ import annotations

import pandas as pd

from buffmini.stage40.objective import TradabilityConfig, compute_tradability_labels


def test_stage40_tradability_labels_tp_before_sl_and_net() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h", tz="UTC"),
            "close": [100.0, 100.5, 101.0, 101.5, 101.2, 101.4],
            "high": [100.2, 100.8, 101.3, 101.8, 101.4, 101.5],
            "low": [99.8, 100.2, 100.8, 101.2, 100.9, 101.1],
            "open": [100.0, 100.4, 100.9, 101.4, 101.1, 101.3],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000],
        }
    )
    cfg = TradabilityConfig(horizon_bars=3, tp_pct=0.004, sl_pct=0.003, round_trip_cost_pct=0.0005)
    labels = compute_tradability_labels(frame, cfg=cfg)
    assert "tp_before_sl" in labels.columns
    assert "net_return_after_cost" in labels.columns
    assert "tradable" in labels.columns
    assert float(labels["tp_before_sl"].iloc[0]) in {0.0, 1.0}
    assert labels["tradable"].astype(int).sum() > 0

