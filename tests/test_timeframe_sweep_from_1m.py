from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.data.storage import save_parquet


def _load_sweep_module():
    path = Path("scripts/run_timeframe_sweep.py")
    spec = importlib.util.spec_from_file_location("stage11_4_timeframe_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load run_timeframe_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_1m(seed: int, rows: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="1min", tz="UTC")
    close = 200.0 + np.cumsum(rng.normal(0.0, 0.3, size=rows))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.25, size=rows)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.25, size=rows)
    volume = rng.uniform(20.0, 800.0, size=rows)
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


def test_timeframe_sweep_runs_from_1m(tmp_path: Path) -> None:
    module = _load_sweep_module()
    raw_dir = tmp_path / "raw"
    derived_dir = tmp_path / "derived"
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    save_parquet(_synthetic_1m(42), symbol="BTC/USDT", timeframe="1m", data_dir=raw_dir)
    save_parquet(_synthetic_1m(43), symbol="ETH/USDT", timeframe="1m", data_dir=raw_dir)

    config = load_config(Path("configs/default.yaml"))
    summary = module.run_timeframe_sweep(
        config=config,
        symbols=["BTC/USDT", "ETH/USDT"],
        base_timeframe="1m",
        operational_timeframes=["15m", "1h"],
        seed=42,
        data_dir=raw_dir,
        derived_dir=derived_dir,
    )
    rows = summary["rows"]
    assert len(rows) == 2
    timeframes = {row["timeframe"] for row in rows}
    assert timeframes == {"15m", "1h"}
    for row in rows:
        assert float(row["runtime_seconds"]) >= 0.0
        assert np.isfinite(float(row["profit_factor"]))
        assert np.isfinite(float(row["expectancy"]))
        assert np.isfinite(float(row["exp_lcb"]))
        assert np.isfinite(float(row["max_drawdown"]))

