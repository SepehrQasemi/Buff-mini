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
    "include_futures_extras": False,
    "futures_extras": {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "1h",
        "max_fill_gap_bars": 8,
        "funding": {
            "z_windows": [30, 90],
            "trend_window": 24,
            "abs_pctl_window": 4320,
            "extreme_pctl": 0.95,
        },
        "open_interest": {
            "chg_windows": [1, 24],
            "z_window": 30,
            "oi_to_volume_window": 24,
            "overlay": {
                "enabled": False,
                "recent_window_days": 30,
                "max_recent_window_days": 90,
                "clamp_to_available": True,
                "inactive_value": "nan",
            },
        },
    },
}

COST_MODEL_DEFAULTS = {
    "mode": "simple",
    "round_trip_cost_pct": 0.1,
    "v2": {
        "slippage_bps_base": 0.5,
        "slippage_bps_vol_mult": 2.0,
        "spread_bps": 0.5,
        "delay_bars": 0,
        "vol_proxy": "atr_pct",
        "vol_lookback": 14,
        "max_total_bps_per_side": 10.0,
    },
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

STAGE6_DEFAULTS = {
    "enabled": False,
    "regime": {
        "atr_percentile_window": 252,
        "vol_expansion_threshold": 0.80,
        "trend_strength_threshold": 0.010,
        "range_atr_threshold": 0.40,
    },
    "confidence_sizing": {
        "scale": 2.0,
        "multiplier_min": 0.5,
        "multiplier_max": 1.5,
    },
    "dynamic_leverage": {
        "trend_multiplier": 1.2,
        "range_multiplier": 0.9,
        "vol_expansion_multiplier": 0.7,
        "dd_soft_threshold": 0.08,
        "dd_soft_multiplier": 0.8,
        "dd_lookback_bars": 168,
        "max_leverage": 50.0,
        "allowed_levels": [1, 2, 3, 5, 10, 15, 20, 25, 50],
    },
}

STAGE8_DEFAULTS = {
    "enabled": True,
    "walkforward_v2": {
        "train_days": 180,
        "holdout_days": 30,
        "forward_days": 30,
        "step_days": 30,
        "reserve_tail_days": 0,
        "min_trades": 10,
        "min_exposure": 0.01,
        "min_usable_windows": 3,
        "stable_min_median_expectancy": 0.0,
        "stable_min_worst_expectancy": 0.0,
        "stable_min_median_profit_factor": 1.0,
        "stable_min_p05_return": 0.0,
        "stable_max_median_max_drawdown": 0.25,
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
        "initial_capital": None,
    },
}

STAGE9_DEFAULTS = {
    "enabled": False,
    "dsl_lite": {
        "enabled": False,
        "funding_selector_enabled": True,
        "oi_selector_enabled": True,
    },
}

STAGE9_3_DEFAULTS = {
    "enabled": False,
    "report_windows_days": [30, 60, 90],
    "nan_policy": "condition_false",
    "ab_non_corruption": {
        "enabled": True,
        "max_trade_count_delta_pct": 1.0,
        "require_equity_identical_for_non_oi": True,
    },
}

STAGE10_DEFAULTS = {
    "enabled": False,
    "cost_mode": "v2",
    "walkforward_v2": True,
    "regimes": {
        "trend_threshold": 0.010,
        "vol_rank_high": 0.80,
        "vol_rank_low": 0.35,
        "compression_z": -0.8,
        "expansion_z": 1.0,
        "volume_z_high": 1.0,
    },
    "activation": {
        "multiplier_min": 0.9,
        "multiplier_max": 1.1,
        "trend_boost": 1.05,
        "range_boost": 1.03,
        "vol_cut": 0.95,
        "chop_cut": 0.93,
    },
    "signals": {
        "families": [
            "BreakoutRetest",
            "MA_SlopePullback",
            "VolCompressionBreakout",
            "BollingerSnapBack",
            "ATR_DistanceRevert",
            "RangeFade",
        ],
        "defaults": {
            "BreakoutRetest": {"donchian_period": 20, "retest_atr_k": 0.8},
            "MA_SlopePullback": {"slope_min": 0.003, "pullback_atr_k": 1.2},
            "VolCompressionBreakout": {"donchian_period": 20, "compression_z": -0.8, "expansion_z": 0.6},
            "BollingerSnapBack": {"rsi_low": 35, "rsi_high": 65},
            "ATR_DistanceRevert": {"distance_k": 2.0},
            "RangeFade": {"donchian_period": 20, "edge_atr_k": 0.6},
        },
    },
    "exits": {
        "modes": ["fixed_atr", "atr_trailing", "breakeven_1r", "partial_tp", "regime_flip_exit"],
        "trailing_atr_k": 1.5,
        "partial_fraction": 0.5,
    },
    "evaluation": {
        "initial_capital": 10000.0,
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
        "dry_run_rows": 2400,
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

    cost_model = _merge_defaults(COST_MODEL_DEFAULTS, config.get("cost_model", {}))
    mode = str(cost_model["mode"])
    if mode not in {"simple", "v2"}:
        raise ValueError("cost_model.mode must be 'simple' or 'v2'")
    # Backward-compatible behavior: `costs.round_trip_cost_pct` is canonical.
    # Keep cost_model in sync so existing scripts that only edit `costs` continue to work.
    cost_model["round_trip_cost_pct"] = float(costs["round_trip_cost_pct"])

    v2 = cost_model["v2"]
    if float(v2["slippage_bps_base"]) < 0:
        raise ValueError("cost_model.v2.slippage_bps_base must be >= 0")
    if float(v2["slippage_bps_vol_mult"]) < 0:
        raise ValueError("cost_model.v2.slippage_bps_vol_mult must be >= 0")
    if float(v2["spread_bps"]) < 0:
        raise ValueError("cost_model.v2.spread_bps must be >= 0")
    if int(v2["delay_bars"]) < 0:
        raise ValueError("cost_model.v2.delay_bars must be >= 0")
    if str(v2["vol_proxy"]) != "atr_pct":
        raise ValueError("cost_model.v2.vol_proxy must be 'atr_pct'")
    if int(v2["vol_lookback"]) < 1:
        raise ValueError("cost_model.v2.vol_lookback must be >= 1")
    if float(v2["max_total_bps_per_side"]) <= 0:
        raise ValueError("cost_model.v2.max_total_bps_per_side must be > 0")
    config["cost_model"] = cost_model

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
    if not isinstance(data["include_futures_extras"], bool):
        raise ValueError("data.include_futures_extras must be bool")
    futures_extras = data["futures_extras"]
    symbols = futures_extras["symbols"]
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("data.futures_extras.symbols must be a non-empty list")
    allowed_symbols = {"BTC/USDT", "ETH/USDT"}
    parsed_symbols = [str(symbol) for symbol in symbols]
    if any(symbol not in allowed_symbols for symbol in parsed_symbols):
        raise ValueError("data.futures_extras.symbols must contain only BTC/USDT and ETH/USDT")
    if str(futures_extras["timeframe"]) != "1h":
        raise ValueError("data.futures_extras.timeframe must be '1h'")
    if int(futures_extras["max_fill_gap_bars"]) < 0:
        raise ValueError("data.futures_extras.max_fill_gap_bars must be >= 0")

    funding_cfg = futures_extras["funding"]
    z_windows = funding_cfg["z_windows"]
    if not isinstance(z_windows, list) or not z_windows:
        raise ValueError("data.futures_extras.funding.z_windows must be a non-empty list")
    if any(int(value) < 2 for value in z_windows):
        raise ValueError("data.futures_extras.funding.z_windows values must be >= 2")
    if int(funding_cfg["trend_window"]) < 1:
        raise ValueError("data.futures_extras.funding.trend_window must be >= 1")
    if int(funding_cfg["abs_pctl_window"]) < 24:
        raise ValueError("data.futures_extras.funding.abs_pctl_window must be >= 24")
    if not 0 < float(funding_cfg["extreme_pctl"]) <= 1:
        raise ValueError("data.futures_extras.funding.extreme_pctl must be in (0,1]")

    oi_cfg = futures_extras["open_interest"]
    chg_windows = oi_cfg["chg_windows"]
    if not isinstance(chg_windows, list) or not chg_windows:
        raise ValueError("data.futures_extras.open_interest.chg_windows must be a non-empty list")
    if any(int(value) < 1 for value in chg_windows):
        raise ValueError("data.futures_extras.open_interest.chg_windows values must be >= 1")
    if int(oi_cfg["z_window"]) < 2:
        raise ValueError("data.futures_extras.open_interest.z_window must be >= 2")
    if int(oi_cfg["oi_to_volume_window"]) < 1:
        raise ValueError("data.futures_extras.open_interest.oi_to_volume_window must be >= 1")
    overlay_cfg = _merge_defaults(DATA_DEFAULTS["futures_extras"]["open_interest"]["overlay"], oi_cfg.get("overlay", {}))
    if not isinstance(overlay_cfg["enabled"], bool):
        raise ValueError("data.futures_extras.open_interest.overlay.enabled must be bool")
    if int(overlay_cfg["max_recent_window_days"]) < 1:
        raise ValueError("data.futures_extras.open_interest.overlay.max_recent_window_days must be >= 1")
    if int(overlay_cfg["recent_window_days"]) < 1:
        raise ValueError("data.futures_extras.open_interest.overlay.recent_window_days must be >= 1")
    if int(overlay_cfg["recent_window_days"]) > int(overlay_cfg["max_recent_window_days"]):
        raise ValueError(
            "data.futures_extras.open_interest.overlay.recent_window_days must be <= max_recent_window_days"
        )
    if not isinstance(overlay_cfg["clamp_to_available"], bool):
        raise ValueError("data.futures_extras.open_interest.overlay.clamp_to_available must be bool")
    if str(overlay_cfg["inactive_value"]) != "nan":
        raise ValueError("data.futures_extras.open_interest.overlay.inactive_value must be 'nan'")
    oi_cfg["overlay"] = overlay_cfg
    futures_extras["open_interest"] = oi_cfg
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

    stage6 = _merge_defaults(STAGE6_DEFAULTS, evaluation.get("stage6", {}))
    if not isinstance(stage6["enabled"], bool):
        raise ValueError("evaluation.stage6.enabled must be bool")
    regime_cfg = stage6["regime"]
    if int(regime_cfg["atr_percentile_window"]) < 2:
        raise ValueError("evaluation.stage6.regime.atr_percentile_window must be >= 2")
    if not 0 <= float(regime_cfg["vol_expansion_threshold"]) <= 1:
        raise ValueError("evaluation.stage6.regime.vol_expansion_threshold must be between 0 and 1")
    if float(regime_cfg["trend_strength_threshold"]) < 0:
        raise ValueError("evaluation.stage6.regime.trend_strength_threshold must be >= 0")
    if not 0 <= float(regime_cfg["range_atr_threshold"]) <= 1:
        raise ValueError("evaluation.stage6.regime.range_atr_threshold must be between 0 and 1")

    confidence_cfg = stage6["confidence_sizing"]
    if float(confidence_cfg["scale"]) <= 0:
        raise ValueError("evaluation.stage6.confidence_sizing.scale must be > 0")
    if float(confidence_cfg["multiplier_min"]) <= 0:
        raise ValueError("evaluation.stage6.confidence_sizing.multiplier_min must be > 0")
    if float(confidence_cfg["multiplier_max"]) <= 0:
        raise ValueError("evaluation.stage6.confidence_sizing.multiplier_max must be > 0")
    if float(confidence_cfg["multiplier_max"]) < float(confidence_cfg["multiplier_min"]):
        raise ValueError("evaluation.stage6.confidence_sizing.multiplier_max must be >= multiplier_min")

    dynamic_cfg = stage6["dynamic_leverage"]
    for key in ["trend_multiplier", "range_multiplier", "vol_expansion_multiplier", "dd_soft_multiplier"]:
        if float(dynamic_cfg[key]) <= 0:
            raise ValueError(f"evaluation.stage6.dynamic_leverage.{key} must be > 0")
    if not 0 <= float(dynamic_cfg["dd_soft_threshold"]) <= 1:
        raise ValueError("evaluation.stage6.dynamic_leverage.dd_soft_threshold must be between 0 and 1")
    if int(dynamic_cfg["dd_lookback_bars"]) < 1:
        raise ValueError("evaluation.stage6.dynamic_leverage.dd_lookback_bars must be >= 1")
    if float(dynamic_cfg["max_leverage"]) <= 0:
        raise ValueError("evaluation.stage6.dynamic_leverage.max_leverage must be > 0")
    allowed_levels = dynamic_cfg.get("allowed_levels", [])
    if not isinstance(allowed_levels, list) or not allowed_levels:
        raise ValueError("evaluation.stage6.dynamic_leverage.allowed_levels must be a non-empty list")
    parsed_levels = [float(value) for value in allowed_levels]
    if any(level <= 0 for level in parsed_levels):
        raise ValueError("evaluation.stage6.dynamic_leverage.allowed_levels values must be > 0")
    if any(right <= left for left, right in zip(parsed_levels[:-1], parsed_levels[1:], strict=False)):
        raise ValueError("evaluation.stage6.dynamic_leverage.allowed_levels must be strictly increasing")
    evaluation["stage6"] = stage6

    stage8 = _merge_defaults(STAGE8_DEFAULTS, evaluation.get("stage8", {}))
    if not isinstance(stage8["enabled"], bool):
        raise ValueError("evaluation.stage8.enabled must be bool")
    wf_v2 = stage8["walkforward_v2"]
    int_fields = ["train_days", "holdout_days", "forward_days", "step_days", "min_usable_windows", "max_hold_bars"]
    for field in int_fields:
        if int(wf_v2[field]) < 1:
            raise ValueError(f"evaluation.stage8.walkforward_v2.{field} must be >= 1")
    if int(wf_v2["reserve_tail_days"]) < 0:
        raise ValueError("evaluation.stage8.walkforward_v2.reserve_tail_days must be >= 0")
    if float(wf_v2["min_trades"]) < 0:
        raise ValueError("evaluation.stage8.walkforward_v2.min_trades must be >= 0")
    if float(wf_v2["min_exposure"]) < 0:
        raise ValueError("evaluation.stage8.walkforward_v2.min_exposure must be >= 0")
    float(wf_v2["stable_min_median_expectancy"])
    float(wf_v2["stable_min_worst_expectancy"])
    float(wf_v2["stable_min_median_profit_factor"])
    float(wf_v2["stable_min_p05_return"])
    if float(wf_v2["stable_max_median_max_drawdown"]) < 0:
        raise ValueError("evaluation.stage8.walkforward_v2.stable_max_median_max_drawdown must be >= 0")
    if wf_v2["initial_capital"] is not None and float(wf_v2["initial_capital"]) <= 0:
        raise ValueError("evaluation.stage8.walkforward_v2.initial_capital must be > 0 when set")
    evaluation["stage8"] = stage8

    stage9 = _merge_defaults(STAGE9_DEFAULTS, evaluation.get("stage9", {}))
    if not isinstance(stage9["enabled"], bool):
        raise ValueError("evaluation.stage9.enabled must be bool")
    dsl_lite_cfg = stage9["dsl_lite"]
    if not isinstance(dsl_lite_cfg["enabled"], bool):
        raise ValueError("evaluation.stage9.dsl_lite.enabled must be bool")
    if not isinstance(dsl_lite_cfg["funding_selector_enabled"], bool):
        raise ValueError("evaluation.stage9.dsl_lite.funding_selector_enabled must be bool")
    if not isinstance(dsl_lite_cfg["oi_selector_enabled"], bool):
        raise ValueError("evaluation.stage9.dsl_lite.oi_selector_enabled must be bool")
    evaluation["stage9"] = stage9
    stage9_3 = _merge_defaults(STAGE9_3_DEFAULTS, evaluation.get("stage9_3", {}))
    if not isinstance(stage9_3["enabled"], bool):
        raise ValueError("evaluation.stage9_3.enabled must be bool")
    report_windows = stage9_3["report_windows_days"]
    if not isinstance(report_windows, list) or not report_windows:
        raise ValueError("evaluation.stage9_3.report_windows_days must be a non-empty list")
    overlay_max_days = int(config["data"]["futures_extras"]["open_interest"]["overlay"]["max_recent_window_days"])
    for value in report_windows:
        if int(value) < 1 or int(value) > overlay_max_days:
            raise ValueError(
                "evaluation.stage9_3.report_windows_days values must be within [1, max_recent_window_days]"
            )
    if str(stage9_3["nan_policy"]) != "condition_false":
        raise ValueError("evaluation.stage9_3.nan_policy must be 'condition_false'")
    ab_cfg = stage9_3["ab_non_corruption"]
    if not isinstance(ab_cfg["enabled"], bool):
        raise ValueError("evaluation.stage9_3.ab_non_corruption.enabled must be bool")
    if float(ab_cfg["max_trade_count_delta_pct"]) < 0:
        raise ValueError("evaluation.stage9_3.ab_non_corruption.max_trade_count_delta_pct must be >= 0")
    if not isinstance(ab_cfg["require_equity_identical_for_non_oi"], bool):
        raise ValueError("evaluation.stage9_3.ab_non_corruption.require_equity_identical_for_non_oi must be bool")
    evaluation["stage9_3"] = stage9_3

    stage10 = _merge_defaults(STAGE10_DEFAULTS, evaluation.get("stage10", {}))
    if not isinstance(stage10["enabled"], bool):
        raise ValueError("evaluation.stage10.enabled must be bool")
    if str(stage10["cost_mode"]) not in {"simple", "v2"}:
        raise ValueError("evaluation.stage10.cost_mode must be 'simple' or 'v2'")
    if not isinstance(stage10["walkforward_v2"], bool):
        raise ValueError("evaluation.stage10.walkforward_v2 must be bool")

    regimes_cfg = stage10["regimes"]
    if float(regimes_cfg["trend_threshold"]) < 0:
        raise ValueError("evaluation.stage10.regimes.trend_threshold must be >= 0")
    if not 0 <= float(regimes_cfg["vol_rank_high"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.vol_rank_high must be in [0,1]")
    if not 0 <= float(regimes_cfg["vol_rank_low"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.vol_rank_low must be in [0,1]")
    float(regimes_cfg["compression_z"])
    float(regimes_cfg["expansion_z"])
    float(regimes_cfg["volume_z_high"])

    activation_cfg = stage10["activation"]
    if "m_min" in activation_cfg and "multiplier_min" not in activation_cfg:
        activation_cfg["multiplier_min"] = activation_cfg["m_min"]
    if "m_max" in activation_cfg and "multiplier_max" not in activation_cfg:
        activation_cfg["multiplier_max"] = activation_cfg["m_max"]
    if "expansion_cut" in activation_cfg and "vol_cut" not in activation_cfg:
        activation_cfg["vol_cut"] = activation_cfg["expansion_cut"]

    if float(activation_cfg["multiplier_min"]) <= 0:
        raise ValueError("evaluation.stage10.activation.multiplier_min must be > 0")
    if float(activation_cfg["multiplier_max"]) <= 0:
        raise ValueError("evaluation.stage10.activation.multiplier_max must be > 0")
    if float(activation_cfg["multiplier_max"]) < float(activation_cfg["multiplier_min"]):
        raise ValueError("evaluation.stage10.activation.multiplier_max must be >= multiplier_min")
    for key in ["trend_boost", "range_boost", "vol_cut", "chop_cut"]:
        if float(activation_cfg[key]) <= 0:
            raise ValueError(f"evaluation.stage10.activation.{key} must be > 0")

    signals_cfg = stage10["signals"]
    if not isinstance(signals_cfg["families"], list) or not signals_cfg["families"]:
        raise ValueError("evaluation.stage10.signals.families must be a non-empty list")
    if not isinstance(signals_cfg["defaults"], dict):
        raise ValueError("evaluation.stage10.signals.defaults must be a mapping")

    exits_cfg = stage10["exits"]
    if not isinstance(exits_cfg["modes"], list) or not exits_cfg["modes"]:
        raise ValueError("evaluation.stage10.exits.modes must be a non-empty list")
    allowed_exit_modes = {"fixed_atr", "atr_trailing", "breakeven_1r", "partial_tp", "regime_flip_exit"}
    if any(str(mode) not in allowed_exit_modes for mode in exits_cfg["modes"]):
        raise ValueError("evaluation.stage10.exits.modes contains unsupported value")
    if float(exits_cfg["trailing_atr_k"]) <= 0:
        raise ValueError("evaluation.stage10.exits.trailing_atr_k must be > 0")
    if not 0 < float(exits_cfg["partial_fraction"]) <= 1:
        raise ValueError("evaluation.stage10.exits.partial_fraction must be in (0,1]")

    eval_cfg = stage10["evaluation"]
    if float(eval_cfg["initial_capital"]) <= 0:
        raise ValueError("evaluation.stage10.evaluation.initial_capital must be > 0")
    if float(eval_cfg["stop_atr_multiple"]) <= 0:
        raise ValueError("evaluation.stage10.evaluation.stop_atr_multiple must be > 0")
    if float(eval_cfg["take_profit_atr_multiple"]) <= 0:
        raise ValueError("evaluation.stage10.evaluation.take_profit_atr_multiple must be > 0")
    if int(eval_cfg["max_hold_bars"]) < 1:
        raise ValueError("evaluation.stage10.evaluation.max_hold_bars must be >= 1")
    if int(eval_cfg["dry_run_rows"]) < 300:
        raise ValueError("evaluation.stage10.evaluation.dry_run_rows must be >= 300")
    evaluation["stage10"] = stage10

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
