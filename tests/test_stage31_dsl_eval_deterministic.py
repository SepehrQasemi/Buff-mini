from __future__ import annotations

import pandas as pd
import numpy as np

from buffmini.stage31.dsl import DSLStrategy, evaluate_strategy
from buffmini.utils.hashing import stable_hash


def _frame(n: int = 500) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    base = np.linspace(100.0, 120.0, n)
    wave = np.sin(np.linspace(0.0, 40.0, n))
    close = base + wave
    open_ = close - 0.1
    high = close + 0.3
    low = close - 0.3
    volume = 1000.0 + np.cos(np.linspace(0.0, 12.0, n)) * 120.0
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_stage31_dsl_eval_deterministic() -> None:
    frame = _frame()
    strategy = DSLStrategy(
        name="deterministic_test",
        long_expr={
            "op": "and",
            "left": {"op": ">", "left": {"op": "feature", "name": "close"}, "right": {"op": "rolling_mean", "x": {"op": "feature", "name": "close"}, "window": 24}},
            "right": {"op": ">", "left": {"op": "rank", "x": {"op": "feature", "name": "volume"}, "window": 48}, "right": {"op": "const", "value": 0.45}},
        },
        short_expr={
            "op": "and",
            "left": {"op": "<", "left": {"op": "feature", "name": "close"}, "right": {"op": "rolling_mean", "x": {"op": "feature", "name": "close"}, "window": 24}},
            "right": {"op": "<", "left": {"op": "rank", "x": {"op": "feature", "name": "volume"}, "window": 48}, "right": {"op": "const", "value": 0.55}},
        },
    )
    first = evaluate_strategy(strategy, frame)
    second = evaluate_strategy(strategy, frame)
    assert stable_hash(first.tolist(), length=16) == stable_hash(second.tolist(), length=16)
    assert set(first.unique()).issubset({-1, 0, 1})

