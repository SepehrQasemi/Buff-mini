"""Configuration loader, validator, and hasher."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from buffmini.types import ConfigDict
from buffmini.utils.hashing import stable_hash


_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "universe": ("symbols", "timeframe", "start", "end"),
    "costs": ("round_trip_cost_pct", "slippage_pct", "funding_pct_per_day"),
    "risk": (
        "risk_per_trade_pct",
        "max_drawdown_pct",
        "max_daily_loss_pct",
        "max_concurrent_positions",
    ),
    "search": ("candidates", "seed"),
    "evaluation": ("stage0_enabled",),
}


STAGE06_DEFAULTS = {
    "window_months": 36,
    "end_mode": "latest",
}

DATA_DEFAULTS = {
    "backend": "parquet",
}

PORTFOLIO_DEFAULTS = {
    "walkforward": {
        "min_usable_windows": 3,
    }
}

RESULT_THRESHOLD_DEFAULTS = {
    "TierA": {
        "min_exp_lcb_holdout": 0.0,
        "min_effective_edge": 0.0,
        "min_trades_per_month_holdout": 5.0,
        "min_pf_adj_holdout": 1.1,
        "max_drawdown_holdout": 0.15,
        "min_exposure_ratio": 0.02,
    },
    "TierB": {
        "min_exp_lcb_holdout": 0.0,
        "min_effective_edge": 0.0,
        "min_trades_per_month_holdout": 2.0,
        "min_pf_adj_holdout": 1.05,
        "max_drawdown_holdout": 0.20,
        "min_exposure_ratio": 0.02,
    },
    "NearMiss": {
        "min_exp_lcb_holdout": -5.0,
    },
}


STAGE1_DEFAULTS = {
    "enabled": True,
    "candidate_count": 5000,
    "top_k": 100,
    "top_m": 20,
    "stage_a_months": 9,
    "stage_b_months": 24,
    "holdout_months": 9,
    "split_mode": "60_20_20",
    "min_holdout_trades": 50,
    "recent_weight": 2.0,
    "min_validation_exposure_ratio": 0.01,
    "min_validation_active_days": 10.0,
    "target_trades_per_month_holdout": 8.0,
    "low_signal_penalty_weight": 1.0,
    "min_trades_per_month_floor": 2.0,
    "allow_rare_if_high_expectancy": False,
    "rare_expectancy_threshold": 3.0,
    "rare_penalty_relief": 0.1,
    "result_thresholds": RESULT_THRESHOLD_DEFAULTS,
    "promotion_holdout_months": [3, 6, 9, 12],
    "walkforward_splits": 3,
    "early_stop_patience": 1000,
    "min_stage_a_evals": 1000,
    "instability_perturbations": 2,
    "weights": {
        "expectancy": 1.0,
        "log_profit_factor": 1.5,
        "max_drawdown": 1.0,
        "complexity": 0.5,
        "instability": 0.75,
    },
    "search_space": {
        "families": [
            "DonchianBreakout",
            "RSIMeanReversion",
            "TrendPullback",
            "BollingerMeanReversion",
            "RangeBreakoutTrendFilter",
        ],
        "gating_modes": ["none", "vol", "vol+regime"],
        "exit_modes": ["fixed_atr", "breakeven_1r", "trailing_atr", "partial_then_trail"],
        "donchian_periods": [20, 55, 100],
        "ema_pairs": [[20, 200], [50, 200], [50, 100]],
        "rsi_long_entry_min": 20,
        "rsi_long_entry_max": 40,
        "rsi_short_entry_min": 60,
        "rsi_short_entry_max": 80,
        "bollinger_period": 20,
        "bollinger_stds": [1.5, 2.0, 2.5],
        "atr_sl_min": 0.8,
        "atr_sl_max": 2.5,
        "atr_tp_min": 1.0,
        "atr_tp_max": 5.0,
        "trailing_atr_k_min": 1.0,
        "trailing_atr_k_max": 2.5,
        "max_holding_bars": [12, 24, 48, 96],
    },
}


def load_config(path: str | Path) -> ConfigDict:
    """Load a YAML config file and validate its shape and values."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    validate_config(config)
    return config


def validate_config(config: ConfigDict) -> None:
    """Validate required sections, keys, and basic value constraints."""

    for section, keys in _REQUIRED_SCHEMA.items():
        if section not in config:
            msg = f"Missing config section: {section}"
            raise ValueError(msg)
        for key in keys:
            if key not in config[section]:
                msg = f"Missing config key: {section}.{key}"
                raise ValueError(msg)

    universe = config["universe"]
    if not universe["symbols"]:
        raise ValueError("universe.symbols must be non-empty")
    if universe["timeframe"] != "1h":
        raise ValueError("Only 1h timeframe is supported in MVP Phase 1")

    costs = config["costs"]
    _validate_percent_value(costs["round_trip_cost_pct"], "costs.round_trip_cost_pct")
    _validate_fraction(costs["slippage_pct"], "costs.slippage_pct")
    if float(costs["funding_pct_per_day"]) < 0:
        raise ValueError("costs.funding_pct_per_day must be >= 0")

    risk = config["risk"]
    _validate_fraction(risk["risk_per_trade_pct"], "risk.risk_per_trade_pct")
    _validate_fraction(risk["max_drawdown_pct"], "risk.max_drawdown_pct")
    _validate_fraction(risk["max_daily_loss_pct"], "risk.max_daily_loss_pct")
    if int(risk["max_concurrent_positions"]) < 1:
        raise ValueError("risk.max_concurrent_positions must be >= 1")

    search = config["search"]
    if int(search["candidates"]) < 1:
        raise ValueError("search.candidates must be >= 1")
    int(search["seed"])

    data = _merge_defaults(DATA_DEFAULTS, config.get("data", {}))
    if str(data["backend"]) not in {"parquet", "duckdb"}:
        raise ValueError("data.backend must be 'parquet' or 'duckdb'")
    config["data"] = data

    portfolio = _merge_defaults(PORTFOLIO_DEFAULTS, config.get("portfolio", {}))
    if int(portfolio["walkforward"]["min_usable_windows"]) < 1:
        raise ValueError("portfolio.walkforward.min_usable_windows must be >= 1")
    config["portfolio"] = portfolio

    evaluation = config["evaluation"]
    if not isinstance(evaluation["stage0_enabled"], bool):
        raise ValueError("evaluation.stage0_enabled must be bool")

    stage06 = _merge_defaults(STAGE06_DEFAULTS, evaluation.get("stage06", {}))
    if int(stage06["window_months"]) < 1:
        raise ValueError("evaluation.stage06.window_months must be >= 1")
    if stage06["end_mode"] != "latest":
        raise ValueError("evaluation.stage06.end_mode must be 'latest'")
    evaluation["stage06"] = stage06

    stage1 = _merge_defaults(STAGE1_DEFAULTS, evaluation.get("stage1", {}))
    _validate_stage1(stage1)
    evaluation["stage1"] = stage1


def compute_config_hash(config: ConfigDict) -> str:
    """Compute deterministic short hash for config payload."""

    return stable_hash(config)


def _validate_fraction(value: Any, name: str) -> None:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_percent_value(value: Any, name: str) -> None:
    numeric = float(value)
    if not 0 <= numeric <= 100:
        raise ValueError(f"{name} must be between 0 and 100 (percent units)")


def _merge_defaults(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_stage1(stage1: dict[str, Any]) -> None:
    int_fields = [
        "candidate_count",
        "top_k",
        "top_m",
        "stage_a_months",
        "stage_b_months",
        "holdout_months",
        "walkforward_splits",
        "early_stop_patience",
        "min_stage_a_evals",
        "instability_perturbations",
    ]
    for field in int_fields:
        if int(stage1[field]) < 1:
            raise ValueError(f"evaluation.stage1.{field} must be >= 1")

    if int(stage1["min_holdout_trades"]) < 0:
        raise ValueError("evaluation.stage1.min_holdout_trades must be >= 0")
    if str(stage1["split_mode"]) != "60_20_20":
        raise ValueError("evaluation.stage1.split_mode must be '60_20_20'")
    if float(stage1["recent_weight"]) < 0:
        raise ValueError("evaluation.stage1.recent_weight must be >= 0")
    if float(stage1["min_validation_exposure_ratio"]) < 0:
        raise ValueError("evaluation.stage1.min_validation_exposure_ratio must be >= 0")
    if float(stage1["min_validation_active_days"]) < 0:
        raise ValueError("evaluation.stage1.min_validation_active_days must be >= 0")
    if float(stage1["target_trades_per_month_holdout"]) <= 0:
        raise ValueError("evaluation.stage1.target_trades_per_month_holdout must be > 0")
    if float(stage1["low_signal_penalty_weight"]) < 0:
        raise ValueError("evaluation.stage1.low_signal_penalty_weight must be >= 0")
    if float(stage1["min_trades_per_month_floor"]) < 0:
        raise ValueError("evaluation.stage1.min_trades_per_month_floor must be >= 0")
    if not isinstance(stage1["allow_rare_if_high_expectancy"], bool):
        raise ValueError("evaluation.stage1.allow_rare_if_high_expectancy must be bool")
    float(stage1["rare_expectancy_threshold"])
    if not 0 <= float(stage1["rare_penalty_relief"]) <= 1:
        raise ValueError("evaluation.stage1.rare_penalty_relief must be between 0 and 1")
    promotion_months = stage1["promotion_holdout_months"]
    if not isinstance(promotion_months, list) or not promotion_months:
        raise ValueError("evaluation.stage1.promotion_holdout_months must be a non-empty list")
    for value in promotion_months:
        if int(value) < 1:
            raise ValueError("evaluation.stage1.promotion_holdout_months values must be >= 1")

    stage1["result_thresholds"] = _normalize_result_thresholds(stage1["result_thresholds"])
    _validate_result_thresholds(stage1["result_thresholds"])

    if int(stage1["top_k"]) > int(stage1["candidate_count"]):
        raise ValueError("evaluation.stage1.top_k must be <= candidate_count")
    if int(stage1["top_m"]) > int(stage1["top_k"]):
        raise ValueError("evaluation.stage1.top_m must be <= top_k")

    weights = stage1["weights"]
    for key in ["expectancy", "log_profit_factor", "max_drawdown", "complexity", "instability"]:
        if float(weights[key]) < 0:
            raise ValueError(f"evaluation.stage1.weights.{key} must be >= 0")

    search_space = stage1["search_space"]
    if not search_space["families"]:
        raise ValueError("evaluation.stage1.search_space.families must be non-empty")
    if not search_space["gating_modes"]:
        raise ValueError("evaluation.stage1.search_space.gating_modes must be non-empty")
    if not search_space["exit_modes"]:
        raise ValueError("evaluation.stage1.search_space.exit_modes must be non-empty")

    if int(search_space["rsi_long_entry_min"]) > int(search_space["rsi_long_entry_max"]):
        raise ValueError("evaluation.stage1.search_space rsi_long_entry_min > rsi_long_entry_max")
    if int(search_space["rsi_short_entry_min"]) > int(search_space["rsi_short_entry_max"]):
        raise ValueError("evaluation.stage1.search_space rsi_short_entry_min > rsi_short_entry_max")

    float_ranges = [
        ("atr_sl_min", "atr_sl_max"),
        ("atr_tp_min", "atr_tp_max"),
        ("trailing_atr_k_min", "trailing_atr_k_max"),
    ]
    for lo_key, hi_key in float_ranges:
        if float(search_space[lo_key]) > float(search_space[hi_key]):
            raise ValueError(f"evaluation.stage1.search_space {lo_key} > {hi_key}")


def _normalize_result_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(thresholds, dict):
        raise ValueError("evaluation.stage1.result_thresholds must be a mapping")

    has_nested_tiers = any(key in thresholds for key in RESULT_THRESHOLD_DEFAULTS)
    if has_nested_tiers:
        return _merge_defaults(RESULT_THRESHOLD_DEFAULTS, thresholds)

    return _merge_defaults(RESULT_THRESHOLD_DEFAULTS, {"TierA": thresholds})


def _validate_result_thresholds(thresholds: dict[str, Any]) -> None:
    for tier_name in ["TierA", "TierB"]:
        tier_thresholds = thresholds[tier_name]
        float(tier_thresholds["min_exp_lcb_holdout"])
        float(tier_thresholds["min_effective_edge"])
        if float(tier_thresholds["min_trades_per_month_holdout"]) < 0:
            raise ValueError(
                f"evaluation.stage1.result_thresholds.{tier_name}.min_trades_per_month_holdout must be >= 0"
            )
        if float(tier_thresholds["min_pf_adj_holdout"]) < 0:
            raise ValueError(
                f"evaluation.stage1.result_thresholds.{tier_name}.min_pf_adj_holdout must be >= 0"
            )
        if float(tier_thresholds["max_drawdown_holdout"]) < 0:
            raise ValueError(
                f"evaluation.stage1.result_thresholds.{tier_name}.max_drawdown_holdout must be >= 0"
            )
        if float(tier_thresholds["min_exposure_ratio"]) < 0:
            raise ValueError(
                f"evaluation.stage1.result_thresholds.{tier_name}.min_exposure_ratio must be >= 0"
            )

    float(thresholds["NearMiss"]["min_exp_lcb_holdout"])
