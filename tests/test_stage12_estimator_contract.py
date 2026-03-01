from __future__ import annotations

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.perf.stage12_estimator import estimate_stage12_runtime


def test_stage12_estimator_contract_fields() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    bench_metrics = {
        "tfs": ["15m", "1h", "2h", "4h"],
        "symbols": ["BTC/USDT"],
        "second_breakdown": {"backtest": 0.40},
    }
    summary = estimate_stage12_runtime(config=config, bench_metrics=bench_metrics)
    required = {
        "n_timeframes",
        "n_strategies",
        "n_cost_levels",
        "n_walkforward_windows",
        "n_symbols",
        "total_task_count",
        "backtest_time_per_strategy_seconds",
        "estimated_total_seconds",
        "estimated_total_minutes",
        "per_timeframe_seconds",
        "scaling_assumption",
        "recommendation",
    }
    assert required.issubset(summary.keys())
    assert summary["n_timeframes"] == 4
    assert summary["n_symbols"] == 1
    assert summary["total_task_count"] >= 1
    assert summary["estimated_total_seconds"] >= 0.0
    assert summary["recommendation"] in {"safe", "heavy", "extreme"}
