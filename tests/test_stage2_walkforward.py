"""Stage-2.5 walk-forward tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from buffmini.config import compute_config_hash, load_config
from buffmini.data.storage import save_parquet
from buffmini.discovery.funnel import _generate_synthetic_ohlcv
from buffmini.portfolio.walkforward import (
    MethodWindowEvaluation,
    WindowSpec,
    build_forward_windows,
    compute_stability_summary,
    run_stage2_walkforward,
)


def _window(name: str, kind: str, start: str, end: str) -> WindowSpec:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    bars = int(((end_ts - start_ts) / pd.Timedelta(hours=1)) + 1)
    return WindowSpec(
        name=name,
        kind=kind,
        expected_start=start_ts,
        expected_end=end_ts,
        actual_start=start_ts,
        actual_end=end_ts,
        truncated=False,
        enough_data=True,
        bar_count=bars,
        note="",
    )


def _evaluation(window: WindowSpec, profit_factor: float, max_drawdown: float) -> MethodWindowEvaluation:
    metrics = {
        "profit_factor": profit_factor,
        "profit_factor_numeric": profit_factor,
        "profit_factor_clipped": min(float(profit_factor), 5.0),
        "raw_profit_factor": profit_factor,
        "expectancy": 1.0,
        "exp_lcb": 0.5,
        "trade_count_total": 10.0,
        "trades_per_month": 5.0,
        "effective_edge": 2.5,
        "exposure_ratio": 0.1,
        "return_pct": 0.02,
        "max_drawdown": max_drawdown,
        "Sharpe_ratio": 1.0,
        "Sortino_ratio": 1.0,
        "Calmar_ratio": 1.0,
        "CAGR_approx": 0.1,
        "CAGR": 0.1,
        "duration_days": 30.0,
        "window_name": window.name,
        "window_kind": window.kind,
        "window_truncated": window.truncated,
        "window_note": window.note,
        "expected_start": window.expected_start.isoformat(),
        "expected_end": window.expected_end.isoformat(),
        "actual_start": window.actual_start.isoformat() if window.actual_start is not None else None,
        "actual_end": window.actual_end.isoformat() if window.actual_end is not None else None,
        "bar_count": window.bar_count,
        "enough_data": window.enough_data,
        "non_finite_metric_keys": [],
        "usable": True if window.kind == "holdout" else True,
        "exclusion_reasons": [],
        "exclusion_reason": "",
    }
    return MethodWindowEvaluation(
        window=window,
        metrics=metrics,
        avg_corr=0.0,
        effective_n=2.0,
        weight_sum=1.0,
        selected_candidates=["c1", "c2"],
    )


def _prepare_stage_runs(tmp_path: Path) -> tuple[Path, str]:
    root = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, seed in [("BTC/USDT", 7), ("ETH/USDT", 11)]:
        frame = _generate_synthetic_ohlcv(symbol=symbol, start="2024-01-01T00:00:00Z", bars=7000, seed=seed)
        save_parquet(frame=frame, symbol=symbol, timeframe="1h", data_dir=data_dir)

    config = load_config(root / "configs" / "default.yaml")
    config = deepcopy(config)
    config["universe"]["start"] = "2024-01-01T00:00:00Z"
    config_hash = compute_config_hash(config)

    stage1_run_id = "stage1_synth"
    stage1_run_dir = runs_dir / stage1_run_id
    (stage1_run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (stage1_run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (stage1_run_dir / "summary.json").write_text(
        json.dumps({"config_hash": config_hash, "data_hash": "synthetic_data_hash"}, indent=2),
        encoding="utf-8",
    )

    holdout_range = "2024-04-01T00:00:00+00:00..2024-04-30T23:00:00+00:00"
    candidate_payloads = [
        {
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
            "holdout_months_used": 1,
            "effective_edge": 10.0,
        },
        {
            "candidate_id": "c2",
            "strategy_name": "Trend Pullback",
            "strategy_family": "TrendPullback",
            "parameters": {
                "channel_period": 20,
                "ema_fast": 50,
                "ema_slow": 200,
                "rsi_long_entry": 30,
                "rsi_short_entry": 70,
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "atr_sl_multiplier": 1.8,
                "atr_tp_multiplier": 2.5,
                "trailing_atr_k": 1.4,
                "max_holding_bars": 48,
                "regime_gate_long": False,
                "regime_gate_short": False,
            },
            "gating": "vol",
            "exit_mode": "fixed_atr",
            "holdout_months_used": 1,
            "effective_edge": 8.0,
        },
    ]
    for index, payload in enumerate(candidate_payloads, start=1):
        path = stage1_run_dir / "candidates" / f"strategy_{index:02d}_{payload['candidate_id']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (stage1_run_dir / "tier_A_candidates.csv").write_text("candidate_id\nc1\n", encoding="utf-8")
    (stage1_run_dir / "tier_B_candidates.csv").write_text("candidate_id\nc2\n", encoding="utf-8")
    strategies_payload = [
        {"candidate_id": "c1", "metrics_holdout": {"date_range": holdout_range}},
        {"candidate_id": "c2", "metrics_holdout": {"date_range": holdout_range}},
    ]
    (stage1_run_dir / "strategies.json").write_text(json.dumps(strategies_payload, indent=2), encoding="utf-8")

    stage2_run_id = "stage2_synth"
    stage2_run_dir = runs_dir / stage2_run_id
    stage2_run_dir.mkdir(parents=True, exist_ok=True)
    stage2_summary = {
        "run_id": stage2_run_id,
        "stage1_run_id": stage1_run_id,
        "window_modes": ["exact_stage1_holdout"],
        "window_mode_note": "Forward windows start immediately after the exact Stage-1 holdout.",
        "portfolio_methods": {
            "equal": {
                "weights": {"c1": 0.5, "c2": 0.5},
                "selected_candidates": ["c1", "c2"],
                "holdout": {"date_range": holdout_range},
            },
            "vol": {
                "weights": {"c1": 0.6, "c2": 0.4},
                "selected_candidates": ["c1", "c2"],
                "holdout": {"date_range": holdout_range},
            },
            "corr-min": {
                "weights": {"c1": 0.5, "c2": 0.5},
                "selected_candidates": ["c1", "c2"],
                "holdout": {"date_range": holdout_range},
            },
        },
    }
    (stage2_run_dir / "portfolio_summary.json").write_text(json.dumps(stage2_summary, indent=2), encoding="utf-8")
    return runs_dir, str(data_dir)


def test_window_builder_produces_non_overlapping_windows() -> None:
    windows = build_forward_windows(
        holdout_end=pd.Timestamp("2026-01-01T00:00:00Z"),
        available_end=pd.Timestamp("2026-04-15T00:00:00Z"),
        forward_days=30,
        num_windows=3,
    )
    assert windows[0].expected_start > pd.Timestamp("2026-01-01T00:00:00Z")
    assert windows[0].expected_end < windows[1].expected_start
    assert windows[1].expected_end < windows[2].expected_start


def test_forward_windows_start_strictly_after_holdout_end() -> None:
    holdout_end = pd.Timestamp("2026-01-01T09:00:00Z")
    windows = build_forward_windows(
        holdout_end=holdout_end,
        available_end=pd.Timestamp("2026-03-15T09:00:00Z"),
        forward_days=30,
        num_windows=2,
    )
    assert all(window.expected_start > holdout_end for window in windows)


def test_degradation_ratio_computation_correctness() -> None:
    evaluations = [
        _evaluation(_window("Holdout", "holdout", "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"), 2.0, 0.10),
        _evaluation(_window("Forward_1", "forward", "2025-02-01T00:00:00Z", "2025-03-02T23:00:00Z"), 1.5, 0.12),
        _evaluation(_window("Forward_2", "forward", "2025-03-03T00:00:00Z", "2025-04-01T23:00:00Z"), 1.0, 0.15),
        _evaluation(_window("Forward_3", "forward", "2025-04-02T00:00:00Z", "2025-05-01T23:00:00Z"), 1.25, 0.14),
    ]
    evaluations[0].metrics["exp_lcb"] = 2.0
    evaluations[1].metrics["exp_lcb"] = 1.5
    evaluations[2].metrics["exp_lcb"] = 1.0
    evaluations[3].metrics["exp_lcb"] = 1.25
    stability = compute_stability_summary(evaluations, stability_metric="exp_lcb")
    assert stability.forward_median == 1.25
    assert stability.degradation_ratio == 0.625
    assert stability.classification == "UNSTABLE"


def test_classification_requires_minimum_usable_windows() -> None:
    insufficient_evaluations = [
        _evaluation(_window("Holdout", "holdout", "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"), 1.2, 0.10),
        _evaluation(_window("Forward_1", "forward", "2025-02-01T00:00:00Z", "2025-03-02T23:00:00Z"), 1.1, 0.12),
    ]
    stability = compute_stability_summary(insufficient_evaluations)
    assert stability.usable_windows == 1
    assert stability.min_usable_windows == 3
    assert stability.classification == "INSUFFICIENT_DATA"
    assert stability.recommendation == "Improve discovery/search space/exits before leverage"


def test_classification_logic_correctness_with_three_windows() -> None:
    stable_evaluations = [
        _evaluation(_window("Holdout", "holdout", "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"), 1.2, 0.10),
        _evaluation(_window("Forward_1", "forward", "2025-02-01T00:00:00Z", "2025-03-02T23:00:00Z"), 1.1, 0.12),
        _evaluation(_window("Forward_2", "forward", "2025-03-03T00:00:00Z", "2025-04-01T23:00:00Z"), 1.3, 0.11),
        _evaluation(_window("Forward_3", "forward", "2025-04-02T00:00:00Z", "2025-05-01T23:00:00Z"), 1.0, 0.09),
    ]
    unstable_evaluations = [
        _evaluation(_window("Holdout", "holdout", "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"), 1.5, 0.10),
        _evaluation(_window("Forward_1", "forward", "2025-02-01T00:00:00Z", "2025-03-02T23:00:00Z"), 0.8, 0.30),
        _evaluation(_window("Forward_2", "forward", "2025-03-03T00:00:00Z", "2025-04-01T23:00:00Z"), 0.9, 0.25),
        _evaluation(_window("Forward_3", "forward", "2025-04-02T00:00:00Z", "2025-05-01T23:00:00Z"), 1.1, 0.22),
    ]
    assert compute_stability_summary(stable_evaluations).classification == "STABLE"
    assert compute_stability_summary(unstable_evaluations).classification == "UNSTABLE"


def test_weights_sum_to_one_and_correlation_matrix_symmetric(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    run_dir = run_stage2_walkforward(
        stage2_run_id="stage2_synth",
        forward_days=30,
        num_windows=3,
        seed=42,
        runs_dir=runs_dir,
        data_dir=Path(data_dir),
        run_id="wf_once",
    )
    summary = json.loads((run_dir / "walkforward_summary.json").read_text(encoding="utf-8"))
    assert abs(sum(summary["method_summaries"]["equal"]["weights"].values()) - 1.0) < 1e-12

    matrix = pd.read_csv(run_dir / "correlation_matrix_holdout.csv", index_col=0)
    assert matrix.equals(matrix.T)


def test_reserve_forward_days_prevents_holdout_from_consuming_tail(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    stage2_summary_path = runs_dir / "stage2_synth" / "portfolio_summary.json"
    stage2_summary = json.loads(stage2_summary_path.read_text(encoding="utf-8"))
    late_holdout_range = "2024-08-15T00:00:00+00:00..2024-10-10T23:00:00+00:00"
    for payload in stage2_summary["portfolio_methods"].values():
        payload["holdout"]["date_range"] = late_holdout_range
    stage2_summary_path.write_text(json.dumps(stage2_summary, indent=2), encoding="utf-8")

    run_dir = run_stage2_walkforward(
        stage2_run_id="stage2_synth",
        forward_days=7,
        num_windows=3,
        reserve_forward_days=21,
        seed=42,
        runs_dir=runs_dir,
        data_dir=Path(data_dir),
        run_id="wf_reserve",
    )

    summary = json.loads((run_dir / "walkforward_summary.json").read_text(encoding="utf-8"))
    holdout = summary["holdout_window"]
    forward_1 = summary["forward_windows"][0]
    assert holdout["actual_end"] != holdout["expected_end"]
    assert pd.Timestamp(holdout["actual_end"]) < pd.Timestamp(forward_1["expected_start"])


def test_deterministic_output_with_same_seed_on_synthetic_data(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    run_a = run_stage2_walkforward(
        stage2_run_id="stage2_synth",
        forward_days=7,
        num_windows=3,
        reserve_forward_days=21,
        seed=42,
        runs_dir=runs_dir,
        data_dir=Path(data_dir),
        run_id="wf_a",
    )
    run_b = run_stage2_walkforward(
        stage2_run_id="stage2_synth",
        forward_days=7,
        num_windows=3,
        reserve_forward_days=21,
        seed=42,
        runs_dir=runs_dir,
        data_dir=Path(data_dir),
        run_id="wf_b",
    )

    summary_a = json.loads((run_a / "walkforward_summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((run_b / "walkforward_summary.json").read_text(encoding="utf-8"))
    summary_a.pop("run_id", None)
    summary_b.pop("run_id", None)
    assert summary_a == summary_b
