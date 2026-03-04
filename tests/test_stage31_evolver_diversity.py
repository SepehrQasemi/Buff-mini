from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from buffmini.stage31.evolve import EvolverConfig, evolve_strategies, signal_similarity


def _frame(n: int = 680) -> pd.DataFrame:
    ts = pd.date_range("2022-06-01", periods=n, freq="1h", tz="UTC")
    x = np.linspace(0.0, 60.0, n)
    close = 90.0 + np.sin(x) * 1.5 + np.cos(x / 2.5) * 0.8 + x * 0.02
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.08,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": 800.0 + np.sin(x / 4.0) * 70.0,
        }
    )


def test_stage31_evolver_diversity() -> None:
    frame = _frame()
    threshold = 0.85
    elites = evolve_strategies(
        frame=frame,
        features=["open", "high", "low", "close", "volume"],
        cfg=EvolverConfig(
            population_size=50,
            generations=4,
            elite_count=12,
            novelty_similarity_max=threshold,
            seed=11,
        ),
    )
    assert not elites.empty
    assert elites["strategy_id"].nunique() == elites.shape[0]
    signals = elites["signal"].tolist()
    for i, j in itertools.combinations(range(len(signals)), 2):
        sim = signal_similarity(signals[i], signals[j])
        assert sim <= threshold + 1e-9

