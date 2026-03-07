from __future__ import annotations

import pandas as pd

from buffmini.stage48.tradability_learning import Stage48Config, compute_stage48_labels


def test_stage48_labels_do_not_use_values_beyond_horizon() -> None:
    idx = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
    close = pd.Series([100 + i * 0.1 for i in range(20)], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000.0,
        }
    )
    cfg = Stage48Config(horizon_bars=3)
    base = compute_stage48_labels(frame, cfg=cfg)
    mutated = frame.copy()
    mutated.loc[15, "close"] = 9999.0  # outside row-0 horizon (bars 1..3)
    changed = compute_stage48_labels(mutated, cfg=cfg)
    assert float(base.loc[0, "net_return_after_cost"]) == float(changed.loc[0, "net_return_after_cost"])

