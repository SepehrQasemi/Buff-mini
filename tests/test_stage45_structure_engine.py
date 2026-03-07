from __future__ import annotations

import pandas as pd

from buffmini.stage45.part1 import compute_market_structure_engine


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC")
    close = pd.Series([100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_stage45_structure_engine_emits_required_columns() -> None:
    out = compute_market_structure_engine(_bars())
    for key in (
        "higher_high",
        "higher_low",
        "lower_high",
        "lower_low",
        "bos",
        "choch",
        "impulsive_leg",
        "corrective_leg",
        "structural_bias",
    ):
        assert key in out.columns
    assert int(out["bos"].sum()) > 0

