from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage31.evolve import EvolverConfig, evolve_strategies
from buffmini.utils.hashing import stable_hash


def _frame(n: int = 720) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    x = np.linspace(0.0, 80.0, n)
    close = 100.0 + np.sin(x / 3.0) * 2.0 + x * 0.03
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.15,
            "high": close + 0.30,
            "low": close - 0.30,
            "close": close,
            "volume": 1200.0 + np.cos(x / 7.0) * 90.0,
        }
    )


def test_stage31_evolver_reproducible() -> None:
    frame = _frame()
    cfg = EvolverConfig(population_size=40, generations=5, elite_count=10, seed=42, novelty_similarity_max=0.90)
    first = evolve_strategies(frame=frame, features=["open", "high", "low", "close", "volume"], cfg=cfg)
    second = evolve_strategies(frame=frame, features=["open", "high", "low", "close", "volume"], cfg=cfg)
    assert not first.empty
    cols = ["strategy_id", "fitness", "trade_count"]
    assert stable_hash(first.loc[:, cols].to_dict(orient="records"), length=16) == stable_hash(second.loc[:, cols].to_dict(orient="records"), length=16)

