"""Stage-3.1 Monte Carlo tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from buffmini.config import compute_config_hash, load_config
from buffmini.data.storage import save_parquet
from buffmini.discovery.funnel import _generate_synthetic_ohlcv
from buffmini.portfolio.monte_carlo import (
    compute_equity_path_metrics,
    sample_block_indices,
    sample_iid_indices,
    simulate_equity_paths,
    summarize_mc,
    run_stage3_monte_carlo,
)


def _prepare_stage_runs(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, seed in [("BTC/USDT", 7), ("ETH/USDT", 11)]:
        frame = _generate_synthetic_ohlcv(symbol=symbol, start="2024-01-01T00:00:00Z", bars=3500, seed=seed)
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

    holdout_range = "2024-02-01T00:00:00+00:00..2024-04-30T23:00:00+00:00"
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


def test_determinism_same_seed_same_summary() -> None:
    pnls = pd.Series([100.0, -50.0, 25.0, 80.0, -20.0], dtype=float)
    first_paths = simulate_equity_paths(pnls, n_paths=2000, method="block", seed=42, initial_equity=10000.0, block_size_trades=2)
    second_paths = simulate_equity_paths(pnls, n_paths=2000, method="block", seed=42, initial_equity=10000.0, block_size_trades=2)
    first_summary = summarize_mc(first_paths, initial_equity=10000.0, ruin_dd_threshold=0.5)
    second_summary = summarize_mc(second_paths, initial_equity=10000.0, ruin_dd_threshold=0.5)

    pd.testing.assert_frame_equal(first_paths, second_paths)
    assert first_summary == second_summary


def test_iid_bootstrap_returns_correct_length() -> None:
    rng = __import__("numpy").random.default_rng(42)
    indices = sample_iid_indices(n_trades=11, n_paths=5, rng=rng)
    assert indices.shape == (5, 11)


def test_block_bootstrap_returns_contiguous_blocks() -> None:
    rng = __import__("numpy").random.default_rng(42)
    indices = sample_block_indices(n_trades=10, n_paths=3, block_size_trades=3, rng=rng)
    assert indices.shape == (3, 10)
    for row in indices:
        for start in [0, 3, 6]:
            segment = row[start : min(start + 3, len(row))]
            if len(segment) > 1:
                assert all(int(b - a) == 1 for a, b in zip(segment[:-1], segment[1:], strict=False))


def test_ruin_probability_is_bounded_and_summary_finite() -> None:
    pnls = pd.Series([200.0, -400.0, 150.0, -250.0, 300.0, -100.0], dtype=float)
    paths = simulate_equity_paths(pnls, n_paths=2000, method="iid", seed=7, initial_equity=10000.0)
    summary = summarize_mc(paths, initial_equity=10000.0, ruin_dd_threshold=0.5)

    assert 0.0 <= summary["tail_probabilities"]["p_ruin"] <= 1.0
    assert "Infinity" not in json.dumps(summary)
    assert "NaN" not in json.dumps(summary)


def test_equity_path_max_drawdown_on_toy_sequence() -> None:
    metrics = compute_equity_path_metrics([100.0, -50.0, -100.0, 50.0], initial_equity=1000.0)
    assert metrics["final_equity"] == 1000.0
    assert metrics["total_return_pct"] == 0.0
    assert round(metrics["max_drawdown"], 6) == round(150.0 / 1100.0, 6)


def test_stage3_runner_writes_finite_summary(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    run_dir = run_stage3_monte_carlo(
        stage2_run_id="stage2_synth",
        methods=["equal", "vol", "corr-min"],
        bootstrap="block",
        block_size_trades=3,
        n_paths=500,
        initial_equity=10000.0,
        ruin_dd_threshold=0.5,
        seed=42,
        leverage=1.0,
        save_paths=False,
        runs_dir=runs_dir,
        data_dir=data_dir,
        run_id="mc_synth",
    )

    text = (run_dir / "mc_summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
