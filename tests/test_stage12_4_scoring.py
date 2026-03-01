from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage12 import sweep
from buffmini.stage12.sweep import run_stage12_sweep


def _frame(rows: int = 360) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 100.0 + np.sin(np.arange(rows, dtype=float) / 9.0) * 2.0 + np.arange(rows, dtype=float) * 0.01
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "ema_50": pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy(dtype=float),
            "bb_mid_20": pd.Series(close).rolling(20).mean().bfill().to_numpy(dtype=float),
            "atr_14": np.full(rows, 1.0, dtype=float),
            "atr_pct_rank_252": np.clip(np.linspace(0.1, 0.9, rows, dtype=float), 0.0, 1.0),
            "ema_slope_50": np.gradient(pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy(dtype=float)),
        }
    )
    return out


def _raw_signal(frame: pd.DataFrame) -> pd.Series:
    idx = np.arange(len(frame))
    raw = np.where(idx % 22 == 4, 1, np.where(idx % 22 == 14, -1, 0))
    return pd.Series(raw, index=frame.index).shift(1).fillna(0).astype(int)


def _cfg() -> dict:
    return {
        "enabled": True,
        "threshold_grid": [0.2, 0.4, 0.6, 0.8],
        "weight_values": [0.5, 1.0, 1.5],
        "trade_rate_target": {"tpm_min": 2.0, "tpm_max": 40.0},
        "cache": {"enabled": True},
    }


def test_stage12_4_determinism_and_bounded_search() -> None:
    frame = _frame()
    raw = _raw_signal(frame)
    strategy = sweep.StrategyVariant(
        strategy_key="BreakoutRetest",
        strategy_name="Breakout Retest",
        source="stage10",
        signal_builder=lambda _: raw,
    )
    sweep._STAGE12_4_SCORE_CACHE.clear()
    signal_a, meta_a = sweep._stage12_4_scored_signal(
        frame=frame,
        strategy=strategy,
        raw_signal=raw,
        stage12_4_cfg=_cfg(),
        seed=42,
    )
    signal_b, meta_b = sweep._stage12_4_scored_signal(
        frame=frame,
        strategy=strategy,
        raw_signal=raw,
        stage12_4_cfg=_cfg(),
        seed=42,
    )
    pd.testing.assert_series_equal(signal_a, signal_b, check_exact=True)
    assert meta_a["chosen_threshold"] == meta_b["chosen_threshold"]
    assert meta_a["chosen_weights"] == meta_b["chosen_weights"]
    max_evals = len(_cfg()["threshold_grid"]) * (len(_cfg()["weight_values"]) ** 3)
    assert int(meta_a["search_evaluations"]) <= max_evals


def test_stage12_4_does_not_mutate_raw_signal_and_cache_hits() -> None:
    frame = _frame()
    raw = _raw_signal(frame)
    raw_before = raw.copy()
    strategy = sweep.StrategyVariant(
        strategy_key="MA_SlopePullback",
        strategy_name="MA SlopePullback",
        source="stage10",
        signal_builder=lambda _: raw,
    )
    sweep._STAGE12_4_SCORE_CACHE.clear()
    _, meta_first = sweep._stage12_4_scored_signal(
        frame=frame,
        strategy=strategy,
        raw_signal=raw,
        stage12_4_cfg=_cfg(),
        seed=7,
    )
    _, meta_second = sweep._stage12_4_scored_signal(
        frame=frame,
        strategy=strategy,
        raw_signal=raw,
        stage12_4_cfg=_cfg(),
        seed=7,
    )
    pd.testing.assert_series_equal(raw, raw_before, check_exact=True)
    assert meta_first["cache_hit"] is False
    assert meta_second["cache_hit"] is True


def test_stage12_4_summary_schema(tmp_path: Path) -> None:
    cfg = deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg["evaluation"]["stage12"]["include_stage06_baselines"] = False
    cfg["evaluation"]["stage10"]["signals"]["families"] = ["BreakoutRetest"]
    cfg["evaluation"]["stage10"]["signals"]["enabled_families"] = ["BreakoutRetest"]
    cfg["evaluation"]["stage12"]["exits"]["variants"] = ["fixed_atr"]
    cfg["evaluation"]["stage12"]["timeframes"] = ["1h"]
    cfg["evaluation"]["stage12"]["symbols"] = ["BTC/USDT"]
    cfg["evaluation"]["stage12"]["monte_carlo"]["n_paths"] = 500
    cfg["evaluation"]["stage12"]["monte_carlo"]["top_pct"] = 1.0
    cfg["evaluation"]["stage12"]["min_usable_windows_valid"] = 1
    cfg["evaluation"]["stage12_3"]["enabled"] = True
    cfg["evaluation"]["stage12_4"]["enabled"] = True
    result = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    payload = json.loads((result["run_dir"] / "stage12_4_metrics.json").read_text(encoding="utf-8"))
    required = {
        "enabled",
        "status",
        "zero_trade_pct",
        "walkforward_executed_true_pct",
        "MC_trigger_rate",
        "invalid_pct",
        "threshold_distribution",
        "weight_distribution",
        "trade_rate_distribution",
        "cache_hit_rate",
    }
    assert required.issubset(payload.keys())
    assert (tmp_path / "docs" / "stage12_4_report.md").exists()
    assert (tmp_path / "docs" / "stage12_4_summary.json").exists()
