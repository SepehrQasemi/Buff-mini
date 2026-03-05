from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.stage34.evolution import EvolutionConfig, run_evolution
from buffmini.utils.hashing import stable_hash


def _dataset(rows: int = 900) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    x = np.linspace(0, 10 * np.pi, rows)
    close = 100.0 + np.sin(x) * 2.0 + 0.01 * np.arange(rows)
    ret = pd.Series(close).pct_change().fillna(0.0)
    atr = pd.Series(close).diff().abs().rolling(14, min_periods=2).mean().fillna(0.2)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000 + (np.arange(rows) % 17) * 9.0,
            "ret_1": ret,
            "ret_3": ret.rolling(3, min_periods=1).sum(),
            "atr_14": atr,
            "atr_pct": atr / pd.Series(close),
            "ma_dist_20": (pd.Series(close) - pd.Series(close).rolling(20, min_periods=2).mean().bfill()) / pd.Series(close),
            "volume_z_24": (pd.Series(1000 + (np.arange(rows) % 17) * 9.0) - 1072.0) / 35.0,
            "label_primary": np.where(pd.Series(close).shift(-2).fillna(pd.Series(close).iloc[-1]) > pd.Series(close), 1, -1),
            "label_auxiliary": -ret.shift(-1).abs().fillna(0.0),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def test_stage34_evolution_deterministic_and_non_noop(tmp_path: Path) -> None:
    dataset = _dataset()
    pool = ["ret_1", "ret_3", "atr_14", "atr_pct", "ma_dist_20", "volume_z_24"]
    reg_a = tmp_path / "a_registry.json"
    reg_b = tmp_path / "b_registry.json"
    out_a = run_evolution(dataset, feature_pool=pool, registry_path=reg_a, cfg=EvolutionConfig(generations=3, max_models_per_generation=4, exploration_pct=0.25, seed=42))
    out_b = run_evolution(dataset, feature_pool=pool, registry_path=reg_b, cfg=EvolutionConfig(generations=3, max_models_per_generation=4, exploration_pct=0.25, seed=42))
    assert stable_hash(out_a, length=16) == stable_hash(out_b, length=16)
    assert len(out_a["generations"]) == 3
    gen0 = out_a["generations"][0]
    gen1 = out_a["generations"][1]
    sig0 = stable_hash(gen0["best"], length=12)
    sig1 = stable_hash(gen1["best"], length=12)
    assert sig0 != sig1 or bool(gen1.get("improved_vs_prev_best", False))
