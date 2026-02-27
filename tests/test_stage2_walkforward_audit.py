"""Audit-grade Stage-2.7 walk-forward tests."""

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
    assess_window_usability,
    compute_stability_summary,
    normalize_profit_factor,
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


def _evaluation(
    window: WindowSpec,
    *,
    exp_lcb: float,
    drawdown: float,
    trade_count: float = 12.0,
    exposure: float = 0.05,
) -> MethodWindowEvaluation:
    metrics = {
        "profit_factor": 1.2,
        "profit_factor_numeric": 1.2,
        "profit_factor_clipped": 1.2,
        "raw_profit_factor": 1.2,
        "expectancy": exp_lcb,
        "exp_lcb": exp_lcb,
        "trade_count_total": trade_count,
        "trades_per_month": 6.0,
        "effective_edge": exp_lcb * 6.0,
        "exposure_ratio": exposure,
        "return_pct": 0.02,
        "max_drawdown": drawdown,
        "Sharpe_ratio": 1.0,
        "Sortino_ratio": 1.0,
        "Calmar_ratio": 1.0,
        "CAGR_approx": 0.1,
        "CAGR": 0.1,
        "duration_days": 30.0,
        "window_name": window.name,
        "window_kind": window.kind,
        "window_truncated": False,
        "window_note": "",
        "expected_start": window.expected_start.isoformat(),
        "expected_end": window.expected_end.isoformat(),
        "actual_start": window.actual_start.isoformat(),
        "actual_end": window.actual_end.isoformat(),
        "bar_count": window.bar_count,
        "enough_data": True,
        "non_finite_metric_keys": [],
        "usable": window.kind == "holdout" or (trade_count >= 10 and exposure >= 0.01),
        "exclusion_reasons": [] if window.kind == "holdout" or (trade_count >= 10 and exposure >= 0.01) else ["low_evidence"],
        "exclusion_reason": "" if window.kind == "holdout" or (trade_count >= 10 and exposure >= 0.01) else "low_evidence",
    }
    return MethodWindowEvaluation(
        window=window,
        metrics=metrics,
        avg_corr=0.0,
        effective_n=2.0,
        weight_sum=1.0,
        selected_candidates=["c1", "c2"],
    )


def _prepare_stage_runs(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, seed in [("BTC/USDT", 7), ("ETH/USDT", 11)]:
        frame = _generate_synthetic_ohlcv(symbol=symbol, start="2024-01-01T00:00:00Z", bars=7000, seed=seed)
        save_parquet(frame=frame, symbol=symbol, timeframe="1h", data_dir=data_dir)

    config = deepcopy(load_config(root / "configs" / "default.yaml"))
    config["universe"]["start"] = "2024-01-01T00:00:00Z"
    config_hash = compute_config_hash(config)

    stage1_run_dir = runs_dir / "stage1_synth"
    (stage1_run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (stage1_run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (stage1_run_dir / "summary.json").write_text(
        json.dumps({"config_hash": config_hash, "data_hash": "synthetic_data_hash"}, indent=2),
        encoding="utf-8",
    )

    holdout_range = "2024-04-01T00:00:00+00:00..2024-04-30T23:00:00+00:00"
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
        "holdout_months_used": 1,
        "effective_edge": 10.0,
    }
    path = stage1_run_dir / "candidates" / "strategy_01_c1.json"
    path.write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
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
        "window_mode_note": "Forward windows start immediately after the exact Stage-1 holdout.",
        "portfolio_methods": {
            "equal": {"weights": {"c1": 1.0}, "selected_candidates": ["c1"], "holdout": {"date_range": holdout_range}},
            "vol": {"weights": {"c1": 1.0}, "selected_candidates": ["c1"], "holdout": {"date_range": holdout_range}},
            "corr-min": {"weights": {"c1": 1.0}, "selected_candidates": ["c1"], "holdout": {"date_range": holdout_range}},
        },
    }
    (stage2_run_dir / "portfolio_summary.json").write_text(json.dumps(stage2_summary, indent=2), encoding="utf-8")
    return runs_dir, data_dir


def test_windows_with_low_trade_count_are_excluded() -> None:
    metrics = {
        "trade_count_total": 5.0,
        "exposure_ratio": 0.05,
        "bar_count": 100,
        "enough_data": True,
        "non_finite_metric_keys": [],
    }
    usable, reasons = assess_window_usability(metrics, min_forward_trades=10, min_forward_exposure=0.01, window_kind="forward")
    assert usable is False
    assert "trade_count<10" in reasons


def test_inf_pf_is_clipped() -> None:
    raw_pf, clipped_pf, is_finite = normalize_profit_factor(float("inf"), pf_clip_max=5.0)
    assert raw_pf == "inf"
    assert clipped_pf == 5.0
    assert is_finite is False


def test_classification_insufficient_when_windows_below_threshold() -> None:
    evaluations = [
        _evaluation(_window("Holdout", "holdout", "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"), exp_lcb=2.0, drawdown=0.1),
        _evaluation(_window("Forward_1", "forward", "2025-02-01T00:00:00Z", "2025-03-02T23:00:00Z"), exp_lcb=1.5, drawdown=0.1),
    ]
    stability = compute_stability_summary(evaluations, min_usable_windows=3, stability_metric="exp_lcb")
    assert stability.classification == "INSUFFICIENT_DATA"


def test_degradation_ratio_uses_median() -> None:
    evaluations = [
        _evaluation(_window("Holdout", "holdout", "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"), exp_lcb=2.0, drawdown=0.1),
        _evaluation(_window("Forward_1", "forward", "2025-02-01T00:00:00Z", "2025-03-02T23:00:00Z"), exp_lcb=1.0, drawdown=0.1),
        _evaluation(_window("Forward_2", "forward", "2025-03-03T00:00:00Z", "2025-04-01T23:00:00Z"), exp_lcb=2.0, drawdown=0.1),
        _evaluation(_window("Forward_3", "forward", "2025-04-02T00:00:00Z", "2025-05-01T23:00:00Z"), exp_lcb=3.0, drawdown=0.1),
    ]
    stability = compute_stability_summary(evaluations, stability_metric="exp_lcb")
    assert stability.forward_median == 2.0
    assert stability.degradation_ratio == 1.0


def test_no_overlap_enforcement_works() -> None:
    holdout_end = pd.Timestamp("2026-01-01T09:00:00Z")
    from buffmini.portfolio.walkforward import build_forward_windows

    windows = build_forward_windows(
        holdout_end=holdout_end,
        available_end=pd.Timestamp("2026-02-01T09:00:00Z"),
        forward_days=7,
        num_windows=3,
    )
    assert windows[0].expected_start > holdout_end
    assert windows[0].expected_end < windows[1].expected_start


def test_no_inf_or_nan_in_summary_json(tmp_path: Path) -> None:
    runs_dir, data_dir = _prepare_stage_runs(tmp_path)
    run_dir = run_stage2_walkforward(
        stage2_run_id="stage2_synth",
        forward_days=7,
        num_windows=3,
        reserve_forward_days=21,
        seed=42,
        runs_dir=runs_dir,
        data_dir=data_dir,
        run_id="wf_audit",
    )
    text = (run_dir / "walkforward_summary.json").read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
