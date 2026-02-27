"""Configuration loader, validator, and hasher."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from buffmini.types import ConfigDict
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import parse_utc_timestamp


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

EXECUTION_DEFAULTS = {
    "mode": "net",
    "per_symbol_netting": True,
    "allow_opposite_signals": False,
    "symbol_scope": "per_symbol",
}

RISK_STAGE4_DEFAULTS = {
    "max_gross_exposure": 5.0,
    "max_net_exposure_per_symbol": 5.0,
    "max_open_positions": 10,
    "sizing": {
        "mode": "risk_budget",
        "risk_per_trade_pct": 1.0,
        "fixed_fraction_pct": 10.0,
    },
    "killswitch": {
        "enabled": True,
        "max_daily_loss_pct": 5.0,
        "max_peak_to_valley_dd_pct": 20.0,
        "max_consecutive_losses": 8,
        "cool_down_bars": 48,
    },
    "reeval": {
        "cadence": "weekly",
        "min_new_bars": 168,
    },
}

STAGE4_DEFAULTS = {
    "source_stage2_run_id": "",
    "source_stage3_3_run_id": "",
    "default_method": "equal",
    "default_leverage": 1.0,
    "output": {
        "write_docs_summary": True,
        "write_run_artifacts": True,
    },
}

UI_STAGE5_DEFAULTS = {
    "stage5": {
        "presets": {
            "quick": {
                "candidate_count": 1000,
                "run_stage4_simulate": 0,
            },
            "full": {
                "candidate_count": 5000,
                "run_stage4_simulate": 0,
            },
        },
        "window_months_options": [3, 6, 12, 36],
    },
}

PORTFOLIO_DEFAULTS = {
    "walkforward": {
        "min_usable_windows": 3,
        "min_forward_trades": 10,
        "min_forward_exposure": 0.01,
        "pf_clip_max": 5.0,
        "stability_metric": "exp_lcb",
    },
    "leverage_selector": {
        "methods": ["equal", "vol"],
        "leverage_levels": [1, 2, 3, 5, 10, 15, 20, 25, 50],
        "bootstrap": "block",
        "block_size_trades": 10,
        "n_paths": 20000,
        "seed": 42,
        "initial_equity": 10000,
        "ruin_dd_threshold": 0.5,
        "constraints": {
            "max_p_ruin": 0.01,
            "max_dd_p95": 0.25,
            "min_return_p05": 0.0,
        },
        "utility": {
            "objective": "expected_log_growth",
            "epsilon": 1e-12,
            "use_final_equity": True,
            "penalize_dd": False,
        },
        "reporting": {
            "include_per_leverage_tables": True,
            "include_binding_constraints": True,
        },
    },
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
    if universe.get("resolved_end_ts") is not None and str(universe.get("resolved_end_ts")).strip() != "":
        try:
            resolved_end_ts = parse_utc_timestamp(str(universe["resolved_end_ts"]))
        except Exception as exc:
            raise ValueError(f"universe.resolved_end_ts must be a valid UTC timestamp: {exc}") from exc
        end_value = universe.get("end")
        if end_value is not None and str(end_value).strip() != "":
            parsed_end = parse_utc_timestamp(str(end_value))
            if resolved_end_ts != parsed_end:
                raise ValueError("universe.end and universe.resolved_end_ts must match when both are set")

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
    risk = _merge_defaults(RISK_STAGE4_DEFAULTS, risk)
    if float(risk["max_gross_exposure"]) <= 0:
        raise ValueError("risk.max_gross_exposure must be > 0")
    if float(risk["max_net_exposure_per_symbol"]) <= 0:
        raise ValueError("risk.max_net_exposure_per_symbol must be > 0")
    if int(risk["max_open_positions"]) < 1:
        raise ValueError("risk.max_open_positions must be >= 1")

    sizing = risk["sizing"]
    if str(sizing["mode"]) not in {"risk_budget", "fixed_fraction"}:
        raise ValueError("risk.sizing.mode must be 'risk_budget' or 'fixed_fraction'")
    _validate_percent_value(sizing["risk_per_trade_pct"], "risk.sizing.risk_per_trade_pct")
    if float(sizing["risk_per_trade_pct"]) <= 0:
        raise ValueError("risk.sizing.risk_per_trade_pct must be > 0")
    _validate_percent_value(sizing["fixed_fraction_pct"], "risk.sizing.fixed_fraction_pct")
    if float(sizing["fixed_fraction_pct"]) <= 0:
        raise ValueError("risk.sizing.fixed_fraction_pct must be > 0")

    killswitch = risk["killswitch"]
    if not isinstance(killswitch["enabled"], bool):
        raise ValueError("risk.killswitch.enabled must be bool")
    _validate_percent_value(killswitch["max_daily_loss_pct"], "risk.killswitch.max_daily_loss_pct")
    if float(killswitch["max_daily_loss_pct"]) <= 0:
        raise ValueError("risk.killswitch.max_daily_loss_pct must be > 0")
    _validate_percent_value(killswitch["max_peak_to_valley_dd_pct"], "risk.killswitch.max_peak_to_valley_dd_pct")
    if float(killswitch["max_peak_to_valley_dd_pct"]) <= 0:
        raise ValueError("risk.killswitch.max_peak_to_valley_dd_pct must be > 0")
    if int(killswitch["max_consecutive_losses"]) < 1:
        raise ValueError("risk.killswitch.max_consecutive_losses must be >= 1")
    if int(killswitch["cool_down_bars"]) < 1:
        raise ValueError("risk.killswitch.cool_down_bars must be >= 1")

    reeval = risk["reeval"]
    if str(reeval["cadence"]) not in {"daily", "weekly", "monthly"}:
        raise ValueError("risk.reeval.cadence must be one of: daily, weekly, monthly")
    if int(reeval["min_new_bars"]) < 1:
        raise ValueError("risk.reeval.min_new_bars must be >= 1")
    config["risk"] = risk

    execution = _merge_defaults(EXECUTION_DEFAULTS, config.get("execution", {}))
    if str(execution["mode"]) not in {"net", "hedge", "isolated"}:
        raise ValueError("execution.mode must be one of: net, hedge, isolated")
    if not isinstance(execution["per_symbol_netting"], bool):
        raise ValueError("execution.per_symbol_netting must be bool")
    if not isinstance(execution["allow_opposite_signals"], bool):
        raise ValueError("execution.allow_opposite_signals must be bool")
    if str(execution["symbol_scope"]) != "per_symbol":
        raise ValueError("execution.symbol_scope must be 'per_symbol'")
    config["execution"] = execution

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
    if int(portfolio["walkforward"]["min_forward_trades"]) < 0:
        raise ValueError("portfolio.walkforward.min_forward_trades must be >= 0")
    if float(portfolio["walkforward"]["min_forward_exposure"]) < 0:
        raise ValueError("portfolio.walkforward.min_forward_exposure must be >= 0")
    if float(portfolio["walkforward"]["pf_clip_max"]) <= 0:
        raise ValueError("portfolio.walkforward.pf_clip_max must be > 0")
    if str(portfolio["walkforward"]["stability_metric"]) not in {"exp_lcb", "effective_edge", "pf_clipped"}:
        raise ValueError("portfolio.walkforward.stability_metric must be one of: exp_lcb, effective_edge, pf_clipped")

    leverage_selector = portfolio["leverage_selector"]
    methods = leverage_selector["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("portfolio.leverage_selector.methods must be a non-empty list")
    for method in methods:
        if str(method) not in {"equal", "vol", "corr-min"}:
            raise ValueError("portfolio.leverage_selector.methods values must be in: equal, vol, corr-min")

    leverage_levels = leverage_selector["leverage_levels"]
    if not isinstance(leverage_levels, list) or not leverage_levels:
        raise ValueError("portfolio.leverage_selector.leverage_levels must be a non-empty list")
    parsed_levels = [float(value) for value in leverage_levels]
    if any(level <= 0 for level in parsed_levels):
        raise ValueError("portfolio.leverage_selector.leverage_levels values must be > 0")
    if any(right <= left for left, right in zip(parsed_levels[:-1], parsed_levels[1:], strict=False)):
        raise ValueError("portfolio.leverage_selector.leverage_levels must be strictly increasing")
    if str(leverage_selector["bootstrap"]) not in {"iid", "block"}:
        raise ValueError("portfolio.leverage_selector.bootstrap must be 'iid' or 'block'")
    if int(leverage_selector["block_size_trades"]) < 1:
        raise ValueError("portfolio.leverage_selector.block_size_trades must be >= 1")
    if int(leverage_selector["n_paths"]) < 1000:
        raise ValueError("portfolio.leverage_selector.n_paths must be >= 1000")
    int(leverage_selector["seed"])
    if float(leverage_selector["initial_equity"]) <= 0:
        raise ValueError("portfolio.leverage_selector.initial_equity must be > 0")
    if not 0 < float(leverage_selector["ruin_dd_threshold"]) < 1:
        raise ValueError("portfolio.leverage_selector.ruin_dd_threshold must be between 0 and 1")

    constraints = leverage_selector["constraints"]
    if not 0 <= float(constraints["max_p_ruin"]) <= 1:
        raise ValueError("portfolio.leverage_selector.constraints.max_p_ruin must be between 0 and 1")
    if float(constraints["max_dd_p95"]) < 0:
        raise ValueError("portfolio.leverage_selector.constraints.max_dd_p95 must be >= 0")
    float(constraints["min_return_p05"])

    utility = leverage_selector["utility"]
    if str(utility["objective"]) != "expected_log_growth":
        raise ValueError("portfolio.leverage_selector.utility.objective must be 'expected_log_growth'")
    if float(utility["epsilon"]) <= 0:
        raise ValueError("portfolio.leverage_selector.utility.epsilon must be > 0")
    if not isinstance(utility["use_final_equity"], bool):
        raise ValueError("portfolio.leverage_selector.utility.use_final_equity must be bool")
    if not isinstance(utility["penalize_dd"], bool):
        raise ValueError("portfolio.leverage_selector.utility.penalize_dd must be bool")

    reporting = leverage_selector["reporting"]
    if not isinstance(reporting["include_per_leverage_tables"], bool):
        raise ValueError("portfolio.leverage_selector.reporting.include_per_leverage_tables must be bool")
    if not isinstance(reporting["include_binding_constraints"], bool):
        raise ValueError("portfolio.leverage_selector.reporting.include_binding_constraints must be bool")

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
    stage4 = _merge_defaults(STAGE4_DEFAULTS, evaluation.get("stage4", {}))
    if not isinstance(stage4["source_stage2_run_id"], str):
        raise ValueError("evaluation.stage4.source_stage2_run_id must be string")
    if not isinstance(stage4["source_stage3_3_run_id"], str):
        raise ValueError("evaluation.stage4.source_stage3_3_run_id must be string")
    if str(stage4["default_method"]) not in {"equal", "vol", "corr-min"}:
        raise ValueError("evaluation.stage4.default_method must be one of: equal, vol, corr-min")
    if float(stage4["default_leverage"]) <= 0:
        raise ValueError("evaluation.stage4.default_leverage must be > 0")
    output_cfg = stage4["output"]
    if not isinstance(output_cfg["write_docs_summary"], bool):
        raise ValueError("evaluation.stage4.output.write_docs_summary must be bool")
    if not isinstance(output_cfg["write_run_artifacts"], bool):
        raise ValueError("evaluation.stage4.output.write_run_artifacts must be bool")
    evaluation["stage4"] = stage4

    ui = _merge_defaults(UI_STAGE5_DEFAULTS, config.get("ui", {}))
    stage5_ui = ui.get("stage5", {})
    presets = stage5_ui["presets"]
    for preset_name in ["quick", "full"]:
        preset = presets[preset_name]
        if int(preset["candidate_count"]) < 1:
            raise ValueError(f"ui.stage5.presets.{preset_name}.candidate_count must be >= 1")
        if int(preset["run_stage4_simulate"]) not in {0, 1}:
            raise ValueError(f"ui.stage5.presets.{preset_name}.run_stage4_simulate must be 0 or 1")

    options = stage5_ui["window_months_options"]
    if not isinstance(options, list) or not options:
        raise ValueError("ui.stage5.window_months_options must be a non-empty list")
    allowed_options = {3, 6, 12, 36}
    parsed_options = {int(value) for value in options}
    if not parsed_options.issubset(allowed_options):
        raise ValueError("ui.stage5.window_months_options values must be a subset of [3, 6, 12, 36]")
    config["ui"] = ui


def compute_config_hash(config: ConfigDict) -> str:
    """Compute deterministic short hash for config payload."""

    return stable_hash(config)


def get_universe_end(config: ConfigDict) -> str | None:
    """Return effective universe end timestamp, preferring pinned resolved_end_ts."""

    universe = config.get("universe", {}) if isinstance(config, dict) else {}
    resolved = universe.get("resolved_end_ts")
    if resolved is not None and str(resolved).strip() != "":
        return str(resolved)
    end = universe.get("end")
    if end is None or str(end).strip() == "":
        return None
    return str(end)


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
