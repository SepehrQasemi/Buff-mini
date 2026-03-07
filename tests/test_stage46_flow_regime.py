from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage46.part2 import compute_flow_regime_engine


def test_stage46_flow_regime_emits_core_columns() -> None:
    idx = pd.date_range("2026-01-01", periods=64, freq="1h", tz="UTC")
    close = pd.Series(100 + np.sin(np.arange(64) / 3.0), dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000 + np.arange(64),
        }
    )
    out = compute_flow_regime_engine(frame)
    for key in (
        "flow_imbalance",
        "imbalance_persistence",
        "flow_burst",
        "flow_exhaustion",
        "flow_confirmed_continuation",
    ):
        assert key in out.columns
    assert out.shape[0] == frame.shape[0]

