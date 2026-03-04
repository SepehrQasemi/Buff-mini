from __future__ import annotations

import pandas as pd

from buffmini.ml.dataset import build_dataset_index
from buffmini.utils.hashing import stable_hash


def _frame(n: int = 512) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    base = pd.Series(range(n), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0 + base * 0.01,
            "high": 100.2 + base * 0.01,
            "low": 99.8 + base * 0.01,
            "close": 100.1 + base * 0.01,
            "volume": 1000.0 + (base % 13),
        }
    )


def test_stage30_dataset_builder_deterministic() -> None:
    frames = {
        "BTC/USDT": _frame(600),
        "ETH/USDT": _frame(620),
    }
    first = build_dataset_index(frames_by_symbol=frames, timeframe="15m", window=128, stride=32)
    second = build_dataset_index(frames_by_symbol=frames, timeframe="15m", window=128, stride=32)
    assert not first.empty
    assert list(first.columns) == list(second.columns)
    assert stable_hash(first.to_dict(orient="records"), length=16) == stable_hash(second.to_dict(orient="records"), length=16)
    assert (first["row_end_exclusive"] - first["row_start"]).eq(128).all()
    assert (first["stride"] == 32).all()

