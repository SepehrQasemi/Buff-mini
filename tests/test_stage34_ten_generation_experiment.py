from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.stage34.evolution import EvolutionConfig, run_evolution
from buffmini.utils.hashing import stable_hash


def _tiny_dataset(rows: int = 420) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
    x = np.linspace(0, 6 * np.pi, rows)
    close = 100.0 + np.sin(x) * 1.8 + 0.02 * np.arange(rows)
    ret = pd.Series(close).pct_change().fillna(0.0)
    atr = pd.Series(close).diff().abs().rolling(14, min_periods=2).mean().fillna(0.1)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.7,
            "low": close - 0.7,
            "close": close,
            "volume": 1000 + (np.arange(rows) % 11) * 8.0,
            "ret_1": ret,
            "ret_3": ret.rolling(3, min_periods=1).sum(),
            "atr_14": atr,
            "atr_pct": atr / pd.Series(close),
            "ma_dist_20": (pd.Series(close) - pd.Series(close).rolling(20, min_periods=2).mean().bfill()) / pd.Series(close),
            "volume_z_24": (pd.Series(1000 + (np.arange(rows) % 11) * 8.0) - 1040.0) / 30.0,
            "label_primary": np.where(pd.Series(close).shift(-2).fillna(pd.Series(close).iloc[-1]) > pd.Series(close), 1, -1),
            "label_auxiliary": -ret.shift(-1).abs().fillna(0.0),
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def test_stage34_ten_generation_output_count_and_determinism(tmp_path: Path) -> None:
    dataset = _tiny_dataset()
    pool = ["ret_1", "ret_3", "atr_14", "atr_pct", "ma_dist_20", "volume_z_24"]
    reg_a = tmp_path / "reg_a.json"
    reg_b = tmp_path / "reg_b.json"
    out_a = run_evolution(
        dataset,
        feature_pool=pool,
        registry_path=reg_a,
        cfg=EvolutionConfig(generations=10, max_models_per_generation=3, exploration_pct=0.34, seed=42),
    )
    out_b = run_evolution(
        dataset,
        feature_pool=pool,
        registry_path=reg_b,
        cfg=EvolutionConfig(generations=10, max_models_per_generation=3, exploration_pct=0.34, seed=42),
    )
    assert len(out_a["generations"]) == 10
    assert len({g["generation"] for g in out_a["generations"]}) == 10
    assert stable_hash(out_a, length=16) == stable_hash(out_b, length=16)
