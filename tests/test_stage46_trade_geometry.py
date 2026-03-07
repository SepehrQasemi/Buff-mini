from __future__ import annotations

import pandas as pd

from buffmini.stage46.part2 import compute_trade_geometry_layer


def test_stage46_trade_geometry_outputs_valid_rr_fields() -> None:
    idx = pd.date_range("2026-01-01", periods=32, freq="1h", tz="UTC")
    close = pd.Series([100 + i * 0.1 for i in range(32)], dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1000.0,
        }
    )
    out = compute_trade_geometry_layer(frame)
    for key in (
        "invalidation_point",
        "stop_distance",
        "target_distance",
        "rr_score",
        "invalidation_quality",
        "structural_preservation_score",
    ):
        assert key in out.columns
    assert float(out["rr_score"].median()) > 1.0

