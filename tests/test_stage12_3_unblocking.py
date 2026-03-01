from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage12.sweep import (
    ExitVariant,
    StrategyVariant,
    _evaluate_combo,
    _resolve_min_trades_required,
    _usable_window,
    run_stage12_sweep,
)


def _synthetic_frame(rows: int = 320) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100.0, 120.0, rows, dtype=float)
    close = base + np.sin(np.arange(rows, dtype=float) / 7.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(rows, 1000.0, dtype=float),
            "atr_14": np.full(rows, 1.0, dtype=float),
            "score_trend": np.where(np.arange(rows) % 2 == 0, 0.8, 0.2).astype(float),
            "score_range": np.where(np.arange(rows) % 2 == 0, 0.2, 0.8).astype(float),
            "score_vol_expansion": np.where(np.arange(rows) % 5 == 0, 0.9, 0.3).astype(float),
            "score_chop": np.where(np.arange(rows) % 7 == 0, 0.6, 0.2).astype(float),
            "atr_pct_rank_252": np.where(np.arange(rows) % 3 == 0, 0.9, 0.2).astype(float),
            "trend_strength_stage10": np.where(np.arange(rows) % 2 == 0, 0.015, 0.005).astype(float),
            "regime_label_stage10": np.where(np.arange(rows) % 2 == 0, "TREND", "RANGE"),
        }
    )


def _custom_signal(frame: pd.DataFrame) -> pd.Series:
    idx = np.arange(len(frame))
    raw = np.where(idx % 25 == 5, 1, np.where(idx % 25 == 15, -1, 0))
    return pd.Series(raw, index=frame.index).shift(1).fillna(0).astype(int)


def test_soft_weighting_does_not_change_trade_timestamps() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    frame = _synthetic_frame()
    strategy = StrategyVariant(
        strategy_key="BreakoutRetest",
        strategy_name="Breakout Retest",
        source="stage10",
        signal_builder=_custom_signal,
    )
    exit_variant = ExitVariant(
        exit_type="fixed_atr",
        engine_exit_mode="fixed_atr",
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        trailing_atr_k=1.5,
        max_hold_bars=24,
    )
    cost_cfg = deepcopy(cfg["cost_model"])
    off = _evaluate_combo(
        frame=frame,
        symbol="BTC/USDT",
        strategy=strategy,
        exit_variant=exit_variant,
        cost_model_cfg=cost_cfg,
        cfg=cfg,
        windows=[],
        min_usable_windows_valid=1,
        context_cfg={},
        stage12_3_cfg={"enabled": False},
        stage12_4_cfg={"enabled": False},
        seed=42,
    )
    on = _evaluate_combo(
        frame=frame,
        symbol="BTC/USDT",
        strategy=strategy,
        exit_variant=exit_variant,
        cost_model_cfg=cost_cfg,
        cfg=cfg,
        windows=[],
        min_usable_windows_valid=1,
        context_cfg={},
        stage12_3_cfg={
            "enabled": True,
            "soft_weights": {
                "enabled": True,
                "min_weight": 0.25,
                "regime_mismatch_weight": 0.5,
                "vol_mismatch_weight": 0.5,
            },
        },
        stage12_4_cfg={"enabled": False},
        seed=42,
    )
    assert off["_trade_timestamps"] == on["_trade_timestamps"]
    assert int(off["trade_count"]) == int(on["trade_count"])


def test_adaptive_usability_reduces_low_usable_rejections() -> None:
    min_disabled = _resolve_min_trades_required(
        trades_per_month=10.0,
        window_days=30,
        base_min_trades=10.0,
        stage12_3_cfg={"enabled": False},
    )
    min_enabled = _resolve_min_trades_required(
        trades_per_month=10.0,
        window_days=30,
        base_min_trades=10.0,
        stage12_3_cfg={
            "enabled": True,
            "usability_adaptive": {"enabled": True, "min_floor": 5, "alpha": 0.35, "max_floor": 80},
        },
    )
    assert min_disabled == 10
    assert min_enabled == 5
    forward_metrics = {"finite": True, "trade_count": 6.0, "exposure_ratio": 0.05}
    usable_disabled, _ = _usable_window(forward_metrics=forward_metrics, min_trades=min_disabled, min_exposure=0.01)
    usable_enabled, _ = _usable_window(forward_metrics=forward_metrics, min_trades=min_enabled, min_exposure=0.01)
    assert usable_disabled is False
    assert usable_enabled is True


def test_stage12_3_metrics_json_has_required_keys(tmp_path: Path) -> None:
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
    result = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    metrics_path = result["run_dir"] / "stage12_3_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = {
        "zero_trade_pct",
        "walkforward_executed_true_pct",
        "MC_trigger_rate",
        "invalid_pct",
        "targets",
        "target_pass_map",
        "passed",
        "status",
    }
    assert required.issubset(payload.keys())
