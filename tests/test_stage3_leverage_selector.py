"""Stage-3.3 leverage selector tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml

from buffmini.config import compute_config_hash, load_config
from buffmini.data.storage import save_parquet
from buffmini.discovery.funnel import _generate_synthetic_ohlcv
from buffmini.portfolio.leverage_selector import (
    choose_best_leverage,
    compute_log_growth,
    run_stage3_leverage_selector,
    summarize_utility,
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


def test_log_utility_computation_matches_expected_values() -> None:
    final_equity = [11000.0, 10000.0, 9000.0]
    expected_individual = [
        compute_log_growth(11000.0, 10000.0, 1e-12),
        compute_log_growth(10000.0, 10000.0, 1e-12),
        compute_log_growth(9000.0, 10000.0, 1e-12),
    ]
    summary = summarize_utility(final_equity, initial_equity=10000.0, epsilon=1e-12)
    assert round(summary["expected_log_growth"], 12) == round(sum(expected_individual) / 3.0, 12)
    assert summary["probability_log_growth_negative"] == 1.0 / 3.0


def test_choose_best_leverage_prefers_max_log_growth_within_feasible_set() -> None:
    cfg = {"constraints": {"max_p_ruin": 0.01, "max_dd_p95": 0.25, "min_return_p05": 0.0}}
    rows = [
        {"leverage": 1.0, "expected_log_growth": 0.02, "return_p05": 0.01, "maxdd_p95": 0.10, "p_ruin": 0.001, "pass_all_constraints": True, "pass_max_p_ruin": True, "pass_max_dd_p95": True, "pass_min_return_p05": True, "failed_constraints": ""},
        {"leverage": 2.0, "expected_log_growth": 0.03, "return_p05": 0.02, "maxdd_p95": 0.15, "p_ruin": 0.005, "pass_all_constraints": True, "pass_max_p_ruin": True, "pass_max_dd_p95": True, "pass_min_return_p05": True, "failed_constraints": ""},
        {"leverage": 3.0, "expected_log_growth": 0.05, "return_p05": -0.01, "maxdd_p95": 0.20, "p_ruin": 0.005, "pass_all_constraints": False, "pass_max_p_ruin": True, "pass_max_dd_p95": True, "pass_min_return_p05": False, "failed_constraints": "min_return_p05"},
    ]
    chosen = choose_best_leverage(rows, cfg)
    assert chosen["status"] == "OK"
    assert chosen["chosen_leverage"] == 2.0


def test_choose_best_leverage_tie_breakers_are_applied_in_order() -> None:
    cfg = {"constraints": {"max_p_ruin": 0.01, "max_dd_p95": 0.25, "min_return_p05": 0.0}}
    rows = [
        {"leverage": 1.0, "expected_log_growth": 0.02, "return_p05": 0.02, "maxdd_p95": 0.10, "p_ruin": 0.005, "pass_all_constraints": True, "pass_max_p_ruin": True, "pass_max_dd_p95": True, "pass_min_return_p05": True, "failed_constraints": ""},
        {"leverage": 2.0, "expected_log_growth": 0.02, "return_p05": 0.02, "maxdd_p95": 0.08, "p_ruin": 0.006, "pass_all_constraints": True, "pass_max_p_ruin": True, "pass_max_dd_p95": True, "pass_min_return_p05": True, "failed_constraints": ""},
        {"leverage": 3.0, "expected_log_growth": 0.02, "return_p05": 0.02, "maxdd_p95": 0.08, "p_ruin": 0.003, "pass_all_constraints": True, "pass_max_p_ruin": True, "pass_max_dd_p95": True, "pass_min_return_p05": True, "failed_constraints": ""},
    ]
    chosen = choose_best_leverage(rows, cfg)
    assert chosen["chosen_leverage"] == 3.0


def test_stage3_selector_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    selector_cfg = deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")["portfolio"]["leverage_selector"])
    selector_cfg["methods"] = ["equal", "vol"]
    selector_cfg["leverage_levels"] = [1, 2, 3]
    selector_cfg["n_paths"] = 2000
    selector_cfg["seed"] = 42

    run_a = run_stage3_leverage_selector(
        stage2_run_id="stage2_synth",
        selector_cfg=selector_cfg,
        runs_dir=runs_dir,
        data_dir=data_dir,
        run_id="selector_a",
    )
    run_b = run_stage3_leverage_selector(
        stage2_run_id="stage2_synth",
        selector_cfg=selector_cfg,
        runs_dir=runs_dir,
        data_dir=data_dir,
        run_id="selector_b",
    )
    summary_a = json.loads((run_a / "selector_summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((run_b / "selector_summary.json").read_text(encoding="utf-8"))

    assert summary_a["overall_choice"] == summary_b["overall_choice"]
    assert summary_a["method_choices"]["equal"]["chosen_leverage"] == summary_b["method_choices"]["equal"]["chosen_leverage"]
    assert summary_a["method_choices"]["vol"]["chosen_leverage"] == summary_b["method_choices"]["vol"]["chosen_leverage"]

