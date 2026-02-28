"""Stage-10.6 sandbox ranking tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.stage10.sandbox import run_stage10_sandbox


def test_sandbox_ranking_is_deterministic(tmp_path: Path) -> None:
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 800
    runs_dir = tmp_path / "runs"

    left = run_stage10_sandbox(
        config=config,
        seed=42,
        dry_run=True,
        cost_mode="v2",
        runs_root=runs_dir,
    )
    right = run_stage10_sandbox(
        config=config,
        seed=42,
        dry_run=True,
        cost_mode="v2",
        runs_root=runs_dir,
    )

    left_rank = pd.read_csv(Path(left["rank_table_path"]))
    right_rank = pd.read_csv(Path(right["rank_table_path"]))
    left_order = left_rank.loc[left_rank["symbol"] == "ALL", ["family", "score"]].reset_index(drop=True)
    right_order = right_rank.loc[right_rank["symbol"] == "ALL", ["family", "score"]].reset_index(drop=True)
    assert left["enabled_signals"] == right["enabled_signals"]
    assert left_order.equals(right_order)


def test_sandbox_outputs_required_artifacts(tmp_path: Path) -> None:
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 700
    summary = run_stage10_sandbox(
        config=config,
        seed=7,
        dry_run=True,
        cost_mode="v2",
        runs_root=tmp_path / "runs",
    )

    run_dir = (tmp_path / "runs") / summary["run_id"]
    rank_path = run_dir / "sandbox_rankings.csv"
    summary_path = run_dir / "sandbox_summary.json"
    per_signal_path = run_dir / "per_signal_metrics.json"
    assert rank_path.exists()
    assert summary_path.exists()
    assert per_signal_path.exists()

    rankings = pd.read_csv(rank_path)
    required_cols = {
        "family",
        "symbol",
        "category",
        "trade_count",
        "trades_per_month",
        "profit_factor",
        "expectancy",
        "max_drawdown",
        "exp_lcb_proxy",
        "drag_penalty",
        "score",
    }
    assert required_cols.issubset(set(rankings.columns))

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(loaded["enabled_signals"], list)
    assert isinstance(loaded["disabled_signals"], list)
