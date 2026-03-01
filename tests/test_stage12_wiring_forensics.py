from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage12.sweep import _stage12_4_scored_signal, run_stage12_sweep


def _base_cfg() -> dict:
    cfg = deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg["evaluation"]["stage12"]["symbols"] = ["BTC/USDT"]
    cfg["evaluation"]["stage12"]["timeframes"] = ["1h"]
    cfg["evaluation"]["stage12"]["exits"]["variants"] = ["fixed_atr"]
    cfg["evaluation"]["stage12"]["monte_carlo"]["n_paths"] = 200
    cfg["evaluation"]["stage12"]["monte_carlo"]["top_pct"] = 1.0
    cfg["evaluation"]["stage12"]["min_usable_windows_valid"] = 1
    cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 300
    return cfg


def test_stage12_trace_has_nonzero_counts_when_enabled(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["evaluation"]["stage12"]["include_stage10_families"] = False
    cfg["evaluation"]["stage12_3"]["enabled"] = True
    cfg["evaluation"]["stage12_4"]["enabled"] = True
    result = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    trace = json.loads((result["run_dir"] / "stage12_trace.json").read_text(encoding="utf-8"))
    assert int(trace["stage12_3"]["combos_seen"]) > 0
    assert int(trace["stage12_3"]["applied_soft_weight_count"]) >= 0
    assert int(trace["stage12_4"]["score_computed_count"]) > 0
    assert int(trace["stage12_4"]["threshold_eval_count"]) > 0


def test_stage12_trace_counts_zero_when_disabled(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["evaluation"]["stage12"]["include_stage10_families"] = False
    cfg["evaluation"]["stage12_3"]["enabled"] = False
    cfg["evaluation"]["stage12_4"]["enabled"] = False
    result = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    trace = json.loads((result["run_dir"] / "stage12_trace.json").read_text(encoding="utf-8"))
    assert int(trace["stage12_3"]["applied_soft_weight_count"]) == 0
    assert int(trace["stage12_4"]["score_computed_count"]) == 0
    assert int(trace["stage12_4"]["threshold_eval_count"]) == 0


def test_stage12_walkforward_executes_on_known_trading_combo(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["evaluation"]["stage12"]["include_stage10_families"] = False
    cfg["evaluation"]["stage12_3"]["enabled"] = True
    cfg["evaluation"]["stage12_4"]["enabled"] = False
    result = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    matrix = pd.read_csv(result["run_dir"] / "stage12_forensic_matrix.csv")
    nonzero = matrix.loc[pd.to_numeric(matrix["trade_count"], errors="coerce").fillna(0.0) > 0.0]
    assert not nonzero.empty
    assert int(nonzero["walkforward_expected_windows"].max()) > 0
    assert bool(nonzero["walkforward_executed"].astype(bool).any())


def test_stage12_4_scoring_changes_kept_ratio_on_fixture() -> None:
    rows = 240
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    close = 100.0 + np.sin(np.arange(rows, dtype=float) / 4.0) * 3.0
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "ema_50": pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy(dtype=float),
            "bb_mid_20": pd.Series(close).rolling(20).mean().bfill().to_numpy(dtype=float),
            "atr_14": np.full(rows, 1.0, dtype=float),
            "atr_pct_rank_252": np.clip(np.linspace(0.1, 0.9, rows), 0.0, 1.0),
            "ema_slope_50": np.gradient(pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy(dtype=float)),
        }
    )
    idx = np.arange(rows)
    raw = pd.Series(np.where(idx % 16 == 4, 1, np.where(idx % 16 == 12, -1, 0)), index=frame.index).shift(1).fillna(0).astype(int)
    signal, meta = _stage12_4_scored_signal(
        frame=frame,
        strategy=type("S", (), {"strategy_key": "BreakoutRetest"})(),
        raw_signal=raw,
        stage12_4_cfg={
            "enabled": True,
            "threshold_grid": [0.3, 0.5, 0.7, 0.9],
            "weight_values": [0.5, 1.0, 1.5],
            "trade_rate_target": {"tpm_min": 2.0, "tpm_max": 40.0},
            "cache": {"enabled": False},
        },
        seed=42,
        trace={
            "stage12_3": {"enabled": False, "applied_soft_weight_count": 0, "combos_seen": 0, "adaptive_usability_samples": [], "fallback_windows_used_count": 0},
            "stage12_4": {"enabled": True, "score_computed_count": 0, "threshold_eval_count": 0, "cache_hit_count": 0, "cache_miss_count": 0},
        },
    )
    raw_count = int((raw != 0).sum())
    kept_count = int((signal != 0).sum())
    assert raw_count > 0
    assert kept_count <= raw_count
    assert kept_count != raw_count
    assert int(meta["search_evaluations"]) > 0
