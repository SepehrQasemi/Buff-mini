"""Stage-10.7 real-data sandbox determinism tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.data.storage import save_parquet
from buffmini.stage10.sandbox import run_stage10_sandbox
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _write_local_series(data_dir: Path, symbol: str, seed: int) -> None:
    frame = synthetic_ohlcv(rows=1200, seed=seed)
    save_parquet(frame=frame, symbol=symbol, timeframe="1h", data_dir=data_dir)


def test_real_data_sandbox_ranking_order_deterministic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_local_series(raw_dir, "BTC/USDT", seed=11)
    _write_local_series(raw_dir, "ETH/USDT", seed=29)

    config = load_config(Path("configs/default.yaml"))
    runs_dir = tmp_path / "runs"

    left = run_stage10_sandbox(
        config=config,
        seed=42,
        dry_run=False,
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        runs_root=runs_dir,
        data_dir=raw_dir,
        derived_dir=tmp_path / "derived",
    )
    right = run_stage10_sandbox(
        config=config,
        seed=42,
        dry_run=False,
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        runs_root=runs_dir,
        data_dir=raw_dir,
        derived_dir=tmp_path / "derived",
    )

    left_rank = pd.read_csv(Path(left["rank_table_path"]))
    right_rank = pd.read_csv(Path(right["rank_table_path"]))
    left_all = left_rank.loc[left_rank["symbol"] == "ALL", ["family", "score"]].reset_index(drop=True)
    right_all = right_rank.loc[right_rank["symbol"] == "ALL", ["family", "score"]].reset_index(drop=True)
    assert left_all.equals(right_all)
    assert left["enabled_signals"] == right["enabled_signals"]
