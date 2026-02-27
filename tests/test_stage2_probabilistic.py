"""Stage-2.8 probabilistic evaluation tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from buffmini.config import compute_config_hash, load_config
from buffmini.data.storage import save_parquet
from buffmini.discovery.funnel import _generate_synthetic_ohlcv
from buffmini.portfolio.probabilistic import (
    aggregate_window_probabilities,
    bootstrap_edge_probability,
    run_stage2_probabilistic,
)
from buffmini.portfolio.rolling_windows import build_rolling_windows


def _prepare_stage_runs(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, seed in [("BTC/USDT", 7), ("ETH/USDT", 11)]:
        frame = _generate_synthetic_ohlcv(symbol=symbol, start="2024-01-01T00:00:00Z", bars=3000, seed=seed)
        save_parquet(frame=frame, symbol=symbol, timeframe="1h", data_dir=data_dir)

    config = deepcopy(load_config(root / "configs" / "default.yaml"))
    config["universe"]["start"] = "2024-01-01T00:00:00Z"
    config["universe"]["end"] = None
    config_hash = compute_config_hash(config)

    stage1_run_dir = runs_dir / "stage1_synth"
    (stage1_run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (stage1_run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (stage1_run_dir / "summary.json").write_text(
        json.dumps({"config_hash": config_hash, "data_hash": "synthetic_data_hash"}, indent=2),
        encoding="utf-8",
    )

    holdout_range = "2024-02-01T00:00:00+00:00..2024-03-31T23:00:00+00:00"
    candidate_payload = {
        "candidate_id": "c1",
        "strategy_name": "Trend Pullback",
        "strategy_family": "TrendPullback",
        "parameters": {
            "channel_period": 55,
            "ema_fast": 20,
            "ema_slow": 200,
            "rsi_long_entry": 35,
            "rsi_short_entry": 65,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 3.0,
            "trailing_atr_k": 1.5,
            "max_holding_bars": 24,
            "regime_gate_long": False,
            "regime_gate_short": False,
        },
        "gating": "none",
        "exit_mode": "fixed_atr",
        "holdout_months_used": 2,
        "effective_edge": 10.0,
    }
    (stage1_run_dir / "candidates" / "strategy_01_c1.json").write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
    (stage1_run_dir / "tier_A_candidates.csv").write_text("candidate_id\nc1\n", encoding="utf-8")
    (stage1_run_dir / "tier_B_candidates.csv").write_text("candidate_id\n", encoding="utf-8")
    (stage1_run_dir / "strategies.json").write_text(
        json.dumps([{"candidate_id": "c1", "metrics_holdout": {"date_range": holdout_range}}], indent=2),
        encoding="utf-8",
    )

    stage2_run_dir = runs_dir / "stage2_synth"
    stage2_run_dir.mkdir(parents=True, exist_ok=True)
    stage2_summary = {
        "run_id": "stage2_synth",
        "stage1_run_id": "stage1_synth",
        "window_modes": ["exact_stage1_holdout"],
        "window_mode_note": "Forward window starts immediately after the exact Stage-1 holdout.",
        "portfolio_methods": {
            "equal": {"weights": {"c1": 1.0}, "selected_candidates": ["c1"], "holdout": {"date_range": holdout_range}},
            "vol": {"weights": {"c1": 1.0}, "selected_candidates": ["c1"], "holdout": {"date_range": holdout_range}},
            "corr-min": {"weights": {"c1": 1.0}, "selected_candidates": ["c1"], "holdout": {"date_range": holdout_range}},
        },
    }
    (stage2_run_dir / "portfolio_summary.json").write_text(json.dumps(stage2_summary, indent=2), encoding="utf-8")
    return runs_dir, data_dir


def test_bootstrap_probability_is_bounded_and_deterministic() -> None:
    pnls = pd.Series([1.0, -0.5, 2.0, 0.25, -0.25], dtype=float)
    first = bootstrap_edge_probability(pnls, n_boot=500, seed=42, pf_clip_max=5.0)
    second = bootstrap_edge_probability(pnls, n_boot=500, seed=42, pf_clip_max=5.0)

    assert first == second
    assert 0.0 <= first["p_edge_gt0"] <= 1.0
    assert 0.0 <= first["p_pf_gt1"] <= 1.0


def test_aggregation_ignores_excluded_windows_and_classifies() -> None:
    strong_rows = [
        {"usable": True, "p_edge_gt0": 0.9, "p_pf_gt1": 0.85, "edge_ci_low": 0.1},
        {"usable": True, "p_edge_gt0": 0.88, "p_pf_gt1": 0.82, "edge_ci_low": 0.09},
        {"usable": True, "p_edge_gt0": 0.91, "p_pf_gt1": 0.87, "edge_ci_low": 0.12},
        {"usable": True, "p_edge_gt0": 0.87, "p_pf_gt1": 0.81, "edge_ci_low": 0.11},
        {"usable": True, "p_edge_gt0": 0.89, "p_pf_gt1": 0.84, "edge_ci_low": 0.1},
        {"usable": False, "p_edge_gt0": 0.01, "p_pf_gt1": 0.01, "edge_ci_low": -5.0},
    ]
    aggregate = aggregate_window_probabilities(strong_rows)

    assert aggregate["usable_windows"] == 5
    assert aggregate["total_windows"] == 6
    assert aggregate["classification"] == "STRONG"
    assert aggregate["p_edge_gt0_min"] > 0.8


def test_rolling_windows_builder_respects_reserve_tail() -> None:
    reserved_tail, windows = build_rolling_windows(
        start_ts=pd.Timestamp("2025-01-01T00:00:00Z"),
        end_ts=pd.Timestamp("2025-12-31T23:00:00Z"),
        window_days=30,
        stride_days=7,
        reserve_tail_days=180,
    )

    assert windows
    assert all(window.start >= reserved_tail.start for window in windows)
    assert all(window.end <= reserved_tail.end for window in windows)
    assert windows[1].start - windows[0].start == pd.Timedelta(days=7)


def test_probabilistic_run_excludes_low_trade_windows(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    run_dir = run_stage2_probabilistic(
        stage2_run_id="stage2_synth",
        window_days=7,
        stride_days=7,
        reserve_tail_days=28,
        min_trades=1_000,
        min_exposure=0.01,
        n_boot=200,
        seed=42,
        runs_dir=runs_dir,
        data_dir=data_dir,
        run_id="prob_excluded",
    )

    excluded = pd.read_csv(run_dir / "excluded_windows.csv")
    summary = json.loads((run_dir / "probabilistic_summary.json").read_text(encoding="utf-8"))

    assert not excluded.empty
    for method_key in ["equal", "vol", "corr-min"]:
        assert summary["method_summaries"][method_key]["aggregate"]["usable_windows"] == 0


def test_probabilistic_summary_has_no_nan_or_inf(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    run_dir = run_stage2_probabilistic(
        stage2_run_id="stage2_synth",
        window_days=7,
        stride_days=7,
        reserve_tail_days=28,
        min_trades=0,
        min_exposure=0.0,
        n_boot=200,
        seed=42,
        runs_dir=runs_dir,
        data_dir=data_dir,
        run_id="prob_summary",
    )

    text = (run_dir / "probabilistic_summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
