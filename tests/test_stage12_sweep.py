from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage12.sweep import run_stage12_sweep, validate_stage12_summary_schema


def _stage12_test_config() -> dict:
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
    return cfg


def test_stage12_summary_schema_contract(tmp_path: Path) -> None:
    cfg = _stage12_test_config()
    result = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    summary = dict(result["summary"])
    validate_stage12_summary_schema(summary)
    assert summary["stage"] == "12"
    assert int(summary["total_combinations"]) == 3
    assert (tmp_path / "docs" / "stage12_report.md").exists()
    assert (tmp_path / "docs" / "stage12_report_summary.json").exists()


def test_stage12_deterministic_metrics(tmp_path: Path) -> None:
    cfg = _stage12_test_config()
    left = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        runs_root=tmp_path / "runs_a",
        docs_dir=tmp_path / "docs_a",
    )
    right = run_stage12_sweep(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframes=["1h"],
        runs_root=tmp_path / "runs_b",
        docs_dir=tmp_path / "docs_b",
    )
    cols = [
        "symbol",
        "timeframe",
        "strategy_key",
        "exit_type",
        "cost_level",
        "exp_lcb",
        "PF",
        "expectancy",
        "maxDD",
        "trades_per_month",
        "stability_classification",
        "cost_sensitivity_slope",
        "robust_score",
        "MC_p_ruin",
        "MC_p_return_negative",
    ]
    left_df = left["leaderboard"].loc[:, cols].sort_values(cols[:5]).reset_index(drop=True)
    right_df = right["leaderboard"].loc[:, cols].sort_values(cols[:5]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left_df, right_df, check_exact=False, atol=1e-12, rtol=0.0)

