from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage31.dsl import DSLStrategy, evaluate_strategy


def _frame(n: int = 600) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    x = np.linspace(0.0, 30.0, n)
    close = 100.0 + np.sin(x) + x * 0.02
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": 900.0 + np.cos(x) * 50.0,
        }
    )


def test_stage31_dsl_no_future_leakage() -> None:
    strategy = DSLStrategy(
        name="leakage_guard",
        long_expr={
            "op": "cross",
            "direction": "up",
            "left": {"op": "feature", "name": "close"},
            "right": {"op": "rolling_mean", "x": {"op": "feature", "name": "close"}, "window": 32},
        },
        short_expr={
            "op": "cross",
            "direction": "down",
            "left": {"op": "feature", "name": "close"},
            "right": {"op": "rolling_mean", "x": {"op": "feature", "name": "close"}, "window": 32},
        },
    )
    base = _frame()
    changed = base.copy()
    changed.loc[350:, "close"] = changed.loc[350:, "close"] * 1.5 + 20.0
    changed.loc[350:, "open"] = changed.loc[350:, "close"] - 0.05
    changed.loc[350:, "high"] = changed.loc[350:, "close"] + 0.10
    changed.loc[350:, "low"] = changed.loc[350:, "close"] - 0.10

    sig_base = evaluate_strategy(strategy, base)
    sig_changed = evaluate_strategy(strategy, changed)

    # Before the changed region, signals must be identical if there is no future leakage.
    assert sig_base.iloc[:300].equals(sig_changed.iloc[:300])

