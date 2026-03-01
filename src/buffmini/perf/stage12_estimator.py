"""Stage-12 runtime preflight estimator."""

from __future__ import annotations

import math
from typing import Any


def estimate_stage12_runtime(config: dict[str, Any], bench_metrics: dict[str, Any]) -> dict[str, Any]:
    """Estimate Stage-12 runtime using structural counts and empirical bench timings."""

    stage12_cfg = config.get("evaluation", {}).get("stage12", {}) if isinstance(config, dict) else {}
    timeframes = stage12_cfg.get("timeframes")
    if not isinstance(timeframes, list) or not timeframes:
        timeframes = list(bench_metrics.get("tfs", []))
    if not timeframes:
        timeframes = [str(config.get("universe", {}).get("operational_timeframe", "1h"))]
    n_timeframes = int(len(timeframes))

    search_space = config.get("evaluation", {}).get("stage1", {}).get("search_space", {})
    families = list(search_space.get("families", []))
    n_strategies = int(len(families)) if families else int(stage12_cfg.get("n_strategies", 1))
    if n_strategies < 1:
        n_strategies = 1

    cost_levels = stage12_cfg.get("cost_levels")
    if isinstance(cost_levels, list) and cost_levels:
        n_cost_levels = int(len(cost_levels))
    else:
        n_cost_levels = int(stage12_cfg.get("n_cost_levels", 2))
    if n_cost_levels < 1:
        n_cost_levels = 1

    n_walkforward_windows = _estimate_walkforward_windows(config=config, stage12_cfg=stage12_cfg)
    bench_symbols = bench_metrics.get("symbols", [])
    if isinstance(bench_symbols, list) and bench_symbols:
        n_symbols = max(1, int(len(bench_symbols)))
    else:
        n_symbols = max(1, int(len(config.get("universe", {}).get("symbols", []))))

    second_breakdown = bench_metrics.get("second_breakdown", {})
    backtest_seconds = float(second_breakdown.get("backtest", 0.0))
    denom = max(1, int(len(bench_metrics.get("tfs", []))) * max(1, int(len(bench_metrics.get("symbols", [])))))
    per_strategy_seconds = float(stage12_cfg.get("backtest_time_per_strategy_seconds", 0.0))
    if per_strategy_seconds <= 0:
        per_strategy_seconds = backtest_seconds / float(denom) if backtest_seconds > 0 else 0.05

    total_tasks = int(n_timeframes * n_strategies * n_cost_levels * n_walkforward_windows * n_symbols)
    estimated_total_seconds = float(total_tasks * per_strategy_seconds)
    estimated_total_minutes = float(estimated_total_seconds / 60.0)

    if estimated_total_minutes > 180:
        recommendation = "extreme"
    elif estimated_total_minutes > 90:
        recommendation = "heavy"
    else:
        recommendation = "safe"

    per_tf_breakdown = {
        str(tf): float(
            n_strategies * n_cost_levels * n_walkforward_windows * n_symbols * per_strategy_seconds
        )
        for tf in timeframes
    }

    return {
        "n_timeframes": n_timeframes,
        "n_strategies": n_strategies,
        "n_cost_levels": n_cost_levels,
        "n_walkforward_windows": n_walkforward_windows,
        "n_symbols": n_symbols,
        "total_task_count": total_tasks,
        "backtest_time_per_strategy_seconds": float(per_strategy_seconds),
        "estimated_total_seconds": estimated_total_seconds,
        "estimated_total_minutes": estimated_total_minutes,
        "per_timeframe_seconds": per_tf_breakdown,
        "scaling_assumption": (
            "Linear scaling by timeframes × strategies × cost levels × walkforward windows × symbols "
            "using empirical backtest seconds from Stage-11.55 bench rerun."
        ),
        "recommendation": recommendation,
    }


def _estimate_walkforward_windows(config: dict[str, Any], stage12_cfg: dict[str, Any]) -> int:
    if "n_walkforward_windows" in stage12_cfg:
        value = int(stage12_cfg["n_walkforward_windows"])
        return max(1, value)

    wf_cfg = (
        config.get("evaluation", {})
        .get("stage8", {})
        .get("walkforward_v2", {})
    )
    train_days = int(wf_cfg.get("train_days", 180))
    holdout_days = int(wf_cfg.get("holdout_days", 30))
    forward_days = int(wf_cfg.get("forward_days", 30))
    step_days = int(wf_cfg.get("step_days", max(holdout_days + forward_days, 30)))
    horizon_days = int(stage12_cfg.get("horizon_days", int(config.get("evaluation", {}).get("stage06", {}).get("window_months", 12)) * 30))

    usable = horizon_days - (train_days + holdout_days + forward_days)
    if usable < 0:
        return 1
    return max(1, int(math.floor(usable / max(1, step_days)) + 1))
