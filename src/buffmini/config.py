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

SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d")
UNIVERSE_DEFAULTS = {
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframe": "1h",
    "base_timeframe": "1h",
    "operational_timeframe": "1h",
    "htf_timeframes": [],
    "start": "2023-01-01T00:00:00Z",
    "end": None,
    "resolved_end_ts": None,
}


STAGE06_DEFAULTS = {
    "window_months": 36,
    "end_mode": "latest",
}

DATA_DEFAULTS = {
    "backend": "parquet",
    "resample_source": "direct",
    "partial_last_bucket": False,
    "feature_cache": {
        "enabled": True,
    },
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
        "trend_rank_strong": 0.60,
        "trend_rank_weak": 0.40,
        "high_vol_rank": 0.75,
        "low_vol_rank": 0.25,
        "chop_flip_window": 48,
        "chop_flip_threshold": 0.18,
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
        "enabled_families": [
            "BreakoutRetest",
            "MA_SlopePullback",
            "ATR_DistanceRevert",
            "RangeFade",
            "VolCompressionBreakout",
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
        "modes": ["fixed_atr", "atr_trailing"],
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
    "sandbox": {
        "top_k_per_category": 2,
        "bootstrap_resamples": 500,
        "exit_mode": "fixed_atr",
    },
}

STAGE11_DEFAULTS = {
    "enabled": False,
    "allow_noop": False,
    "mtf": {
        "base_timeframe": "1h",
        "layers": [
            {
                "name": "htf_4h",
                "timeframe": "4h",
                "role": "context",
                "features": [
                    "ema_50",
                    "ema_200",
                    "ema_slope_50",
                    "atr_14",
                    "atr_pct",
                    "atr_pct_rank_252",
                    "bb_mid_20",
                    "bb_upper_20_2",
                    "bb_lower_20_2",
                    "bb_bandwidth_20",
                    "volume_z_120",
                ],
                "tolerance_bars": 4,
                "enabled": True,
            },
            {
                "name": "ltf_15m",
                "timeframe": "15m",
                "role": "confirm",
                "features": ["ema_50", "ema_slope_50", "atr_pct_rank_252", "volume_z_120"],
                "tolerance_bars": 2,
                "enabled": False,
            },
        ],
        "feature_pack_params": {},
        "hooks_enabled": {"bias": True, "confirm": False, "exit": False},
    },
    "hooks": {
        "bias": {
            "enabled": True,
            "multiplier_min": 0.9,
            "multiplier_max": 1.1,
            "trend_boost": 1.10,
            "range_boost": 1.05,
            "vol_cut": 0.95,
            "trend_slope_scale": 0.01,
        },
        "confirm": {"enabled": False, "threshold": 0.55, "max_delay_bars": 3},
        "exit": {"enabled": False, "tighten_trailing_scale": 0.9},
    },
    "trade_count_guard": {
        "bias_max_drop_pct": 2.0,
        "confirm_max_drop_pct": 25.0,
        "material_pf_improvement": 0.05,
        "material_exp_lcb_improvement": 0.5,
    },
}

STAGE11_55_DEFAULTS = {
    "cache": {
        "max_entries_per_tf": 5,
        "max_total_mb": 2048,
    }
}

STAGE12_DEFAULTS = {
    "enabled": False,
    "base_timeframe": "1m",
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframes": ["15m", "30m", "1h", "2h", "4h", "1d"],
    "include_stage06_baselines": True,
    "include_stage10_families": True,
    "exits": {
        "variants": ["fixed_atr", "structure_trailing", "time_based"],
        "time_based_stop_atr_multiple": 1_000_000.0,
        "time_based_take_profit_atr_multiple": 1_000_000.0,
    },
    "cost_scenarios": {
        "low": {
            "slippage_bps_base": 0.25,
            "slippage_bps_vol_mult": 1.0,
            "spread_bps": 0.25,
            "delay_bars": 0,
        },
        "realistic": {
            "use_config_default": True,
        },
        "high": {
            "slippage_bps_base": 1.5,
            "slippage_bps_vol_mult": 3.0,
            "spread_bps": 1.5,
            "delay_bars": 1,
        },
    },
    "robustness": {
        "instability_penalty": {
            "STABLE": 0.0,
            "UNSTABLE": 0.5,
            "INSUFFICIENT_DATA": 1.0,
        },
        "cost_sensitivity_penalty_weight": 1.0,
    },
    "min_usable_windows_valid": 3,
    "monte_carlo": {
        "enabled": True,
        "top_pct": 0.2,
        "bootstrap": "block",
        "block_size_trades": 10,
        "n_paths": 5000,
        "initial_equity": 10000.0,
        "ruin_dd_threshold": 0.5,
    },
    "forensics": {
        "suspicious_backtest_ms_threshold": 5.0,
        "context_model": {
            "regime_alignment_weight": 0.40,
            "volatility_alignment_weight": 0.25,
            "trend_strength_weight": 0.25,
            "chop_penalty_weight": 0.20,
            "separation_effect_size_threshold": 0.10,
            "min_samples": 30,
        },
    },
}

STAGE12_3_DEFAULTS = {
    "enabled": False,
    "soft_weights": {
        "enabled": True,
        "min_weight": 0.25,
        "regime_mismatch_weight": 0.5,
        "vol_mismatch_weight": 0.5,
    },
    "usability_adaptive": {
        "enabled": True,
        "min_floor": 5,
        "alpha": 0.35,
        "max_floor": 80,
    },
}

STAGE12_4_DEFAULTS = {
    "enabled": False,
    "threshold_grid": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "weight_values": [0.5, 1.0, 1.5],
    "trade_rate_target": {
        "tpm_min": 2.0,
        "tpm_max": 40.0,
    },
    "cache": {
        "enabled": True,
    },
}

STAGE13_DEFAULTS = {
    "enabled": False,
    "seed": 42,
    "families": {
        "enabled": ["price", "volatility", "flow"],
    },
    "composer": {
        "mode": "weighted_sum",
        "weights": {"price": 0.45, "volatility": 0.35, "flow": 0.20},
        "gated": {
            "gate_family": "volatility",
            "gate_threshold": 0.20,
            "entry_threshold": 0.25,
        },
    },
    "gates": {
        "zero_trade_pct_max": 40.0,
        "min_trade_count_ratio_vs_baseline": 0.60,
        "min_walkforward_executed_true_pct": 1.0,
        "min_mc_trigger_rate": 1.0,
    },
    "price": {
        "entry_threshold": 0.30,
        "donchian_period": 20,
        "retest_bars": 6,
        "retest_atr_k": 0.8,
        "trend_pullback_k": 1.2,
        "false_break_lookback": 4,
        "seq_min_len": 2,
        "false_signal_horizon_bars": 6,
        "sweep_grid": {
            "entry_threshold": [0.22, 0.28, 0.34, 0.40],
            "retest_atr_k": [0.6, 0.8, 1.0],
            "trend_pullback_k": [0.8, 1.2, 1.6],
        },
    },
    "volatility": {
        "entry_threshold": 0.28,
        "compression_z": -0.8,
        "expansion_z": 0.8,
        "exhaustion_rank": 0.9,
        "atr_slope_window": 12,
        "vol_wide_stop_mult": 1.10,
        "sweep_grid": {
            "entry_threshold": [0.22, 0.28, 0.34],
            "compression_z": [-1.2, -0.8, -0.4],
            "expansion_z": [0.6, 0.8, 1.0],
        },
    },
    "flow": {
        "entry_threshold": 0.30,
        "volume_window": 48,
        "anomaly_cap": 3.0,
        "riskoff_hard_gate": False,
        "risk_off_penalty": 0.85,
        "sweep_grid": {
            "entry_threshold": [0.25, 0.30, 0.35],
            "anomaly_cap": [2.0, 3.0, 4.0],
        },
    },
}

STAGE14_DEFAULTS = {
    "enabled": False,
    "seed": 42,
    "models": {"allowed": ["logreg_l2", "ridge"]},
    "max_features": 20,
    "trade_rate_bounds": {"min_tpm": 5.0, "max_tpm": 80.0},
    "weighting": {
        "enabled": True,
        "l2_grid": [0.1, 0.3, 1.0, 3.0],
        "coef_clip": 3.0,
        "drift_threshold": 0.40,
    },
    "threshold_calibration": {
        "enabled": True,
        "low_grid": [0.20, 0.25, 0.30],
        "high_grid": [0.35, 0.45, 0.55],
    },
    "nested_walkforward": {"enabled": True, "folds": 3},
    "meta_family": {"enabled": False, "min_families_required": 2},
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

    universe = _merge_defaults(UNIVERSE_DEFAULTS, config["universe"])
    if not universe["symbols"]:
        raise ValueError("universe.symbols must be non-empty")
    timeframe = str(universe["timeframe"]).strip().lower()
    base_timeframe = str(universe.get("base_timeframe") or timeframe).strip().lower()
    operational_timeframe = str(universe.get("operational_timeframe") or timeframe).strip().lower()
    htf_timeframes = [str(value).strip().lower() for value in list(universe.get("htf_timeframes", []))]
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"universe.timeframe must be one of {SUPPORTED_TIMEFRAMES}")
    if base_timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"universe.base_timeframe must be one of {SUPPORTED_TIMEFRAMES}")
    if operational_timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"universe.operational_timeframe must be one of {SUPPORTED_TIMEFRAMES}")
    if timeframe != operational_timeframe:
        raise ValueError("universe.timeframe must match universe.operational_timeframe")
    _validate_timeframe_multiple(
        base_timeframe=base_timeframe,
        child_timeframe=operational_timeframe,
        field_name="universe.operational_timeframe",
    )
    for idx, item in enumerate(htf_timeframes):
        if item not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"universe.htf_timeframes[{idx}] must be one of {SUPPORTED_TIMEFRAMES}")
        _validate_timeframe_multiple(
            base_timeframe=base_timeframe,
            child_timeframe=item,
            field_name=f"universe.htf_timeframes[{idx}]",
        )
    universe["base_timeframe"] = base_timeframe
    universe["operational_timeframe"] = operational_timeframe
    universe["timeframe"] = operational_timeframe
    universe["htf_timeframes"] = htf_timeframes
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
    config["universe"] = universe

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
    if str(data["resample_source"]) not in {"direct", "base"}:
        raise ValueError("data.resample_source must be 'direct' or 'base'")
    if not isinstance(data["partial_last_bucket"], bool):
        raise ValueError("data.partial_last_bucket must be bool")
    feature_cache_cfg = data.get("feature_cache", {})
    if not isinstance(feature_cache_cfg, dict):
        raise ValueError("data.feature_cache must be mapping")
    if not isinstance(feature_cache_cfg.get("enabled", True), bool):
        raise ValueError("data.feature_cache.enabled must be bool")
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
    if "trend_threshold" in regimes_cfg and "trend_rank_strong" not in regimes_cfg:
        regimes_cfg["trend_rank_strong"] = 0.60
    if "vol_rank_high" in regimes_cfg and "high_vol_rank" not in regimes_cfg:
        regimes_cfg["high_vol_rank"] = regimes_cfg["vol_rank_high"]
    if "vol_rank_low" in regimes_cfg and "low_vol_rank" not in regimes_cfg:
        regimes_cfg["low_vol_rank"] = regimes_cfg["vol_rank_low"]
    if "trend_rank_weak" not in regimes_cfg:
        regimes_cfg["trend_rank_weak"] = 0.40
    if "chop_flip_window" not in regimes_cfg:
        regimes_cfg["chop_flip_window"] = 48
    if "chop_flip_threshold" not in regimes_cfg:
        regimes_cfg["chop_flip_threshold"] = 0.18

    if not 0 <= float(regimes_cfg["trend_rank_strong"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.trend_rank_strong must be in [0,1]")
    if not 0 <= float(regimes_cfg["trend_rank_weak"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.trend_rank_weak must be in [0,1]")
    if float(regimes_cfg["trend_rank_strong"]) <= float(regimes_cfg["trend_rank_weak"]):
        raise ValueError("evaluation.stage10.regimes.trend_rank_strong must be > trend_rank_weak")
    if not 0 <= float(regimes_cfg["high_vol_rank"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.high_vol_rank must be in [0,1]")
    if not 0 <= float(regimes_cfg["low_vol_rank"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.low_vol_rank must be in [0,1]")
    if float(regimes_cfg["high_vol_rank"]) <= float(regimes_cfg["low_vol_rank"]):
        raise ValueError("evaluation.stage10.regimes.high_vol_rank must be > low_vol_rank")
    if int(regimes_cfg["chop_flip_window"]) < 4:
        raise ValueError("evaluation.stage10.regimes.chop_flip_window must be >= 4")
    if not 0 <= float(regimes_cfg["chop_flip_threshold"]) <= 1:
        raise ValueError("evaluation.stage10.regimes.chop_flip_threshold must be in [0,1]")
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
    if not isinstance(signals_cfg.get("enabled_families", []), list) or not signals_cfg.get("enabled_families"):
        raise ValueError("evaluation.stage10.signals.enabled_families must be a non-empty list")
    family_set = {str(name) for name in signals_cfg["families"]}
    enabled_set = {str(name) for name in signals_cfg["enabled_families"]}
    if not enabled_set.issubset(family_set):
        raise ValueError("evaluation.stage10.signals.enabled_families must be a subset of families")
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
    sandbox_cfg = stage10.get("sandbox", {})
    if int(sandbox_cfg["top_k_per_category"]) < 1:
        raise ValueError("evaluation.stage10.sandbox.top_k_per_category must be >= 1")
    if int(sandbox_cfg["bootstrap_resamples"]) < 100:
        raise ValueError("evaluation.stage10.sandbox.bootstrap_resamples must be >= 100")
    if str(sandbox_cfg["exit_mode"]) not in {"fixed_atr", "atr_trailing"}:
        raise ValueError("evaluation.stage10.sandbox.exit_mode must be fixed_atr or atr_trailing")
    evaluation["stage10"] = stage10

    stage11 = _merge_defaults(STAGE11_DEFAULTS, evaluation.get("stage11", {}))
    if not isinstance(stage11["enabled"], bool):
        raise ValueError("evaluation.stage11.enabled must be bool")
    if not isinstance(stage11.get("allow_noop", False), bool):
        raise ValueError("evaluation.stage11.allow_noop must be bool")
    mtf_cfg = stage11["mtf"]
    if str(mtf_cfg["base_timeframe"]) != "1h":
        raise ValueError("evaluation.stage11.mtf.base_timeframe must be '1h'")
    layers = mtf_cfg["layers"]
    if not isinstance(layers, list):
        raise ValueError("evaluation.stage11.mtf.layers must be a list")
    allowed_roles = {"context", "confirm", "exit", "features_only"}
    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}] must be mapping")
        if not str(layer.get("name", "")).strip():
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].name must be non-empty")
        timeframe_value = str(layer.get("timeframe", "")).strip().lower()
        if not timeframe_value or timeframe_value[-1] not in {"m", "h", "d"}:
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].timeframe must be like 15m/1h/1d")
        if int(str(timeframe_value[:-1])) <= 0:
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].timeframe must have positive interval")
        if str(layer.get("role", "")) not in allowed_roles:
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].role unsupported")
        features = layer.get("features", [])
        if not isinstance(features, list):
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].features must be list")
        if int(layer.get("tolerance_bars", 1)) < 1:
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].tolerance_bars must be >= 1")
        if not isinstance(layer.get("enabled", True), bool):
            raise ValueError(f"evaluation.stage11.mtf.layers[{idx}].enabled must be bool")
    hooks_enabled = mtf_cfg.get("hooks_enabled", {})
    for key in ("bias", "confirm", "exit"):
        if not isinstance(hooks_enabled.get(key, False), bool):
            raise ValueError(f"evaluation.stage11.mtf.hooks_enabled.{key} must be bool")

    hooks_cfg = stage11["hooks"]
    for key in ("bias", "confirm", "exit"):
        if not isinstance(hooks_cfg[key]["enabled"], bool):
            raise ValueError(f"evaluation.stage11.hooks.{key}.enabled must be bool")
    bias_cfg = hooks_cfg["bias"]
    if float(bias_cfg["multiplier_min"]) <= 0:
        raise ValueError("evaluation.stage11.hooks.bias.multiplier_min must be > 0")
    if float(bias_cfg["multiplier_max"]) <= 0:
        raise ValueError("evaluation.stage11.hooks.bias.multiplier_max must be > 0")
    if float(bias_cfg["multiplier_max"]) < float(bias_cfg["multiplier_min"]):
        raise ValueError("evaluation.stage11.hooks.bias.multiplier_max must be >= multiplier_min")
    for field in ("trend_boost", "range_boost", "vol_cut", "trend_slope_scale"):
        if float(bias_cfg[field]) <= 0:
            raise ValueError(f"evaluation.stage11.hooks.bias.{field} must be > 0")
    confirm_cfg = hooks_cfg["confirm"]
    if not 0 <= float(confirm_cfg["threshold"]) <= 1:
        raise ValueError("evaluation.stage11.hooks.confirm.threshold must be between 0 and 1")
    if int(confirm_cfg["max_delay_bars"]) < 0:
        raise ValueError("evaluation.stage11.hooks.confirm.max_delay_bars must be >= 0")
    exit_cfg = hooks_cfg["exit"]
    if float(exit_cfg["tighten_trailing_scale"]) <= 0:
        raise ValueError("evaluation.stage11.hooks.exit.tighten_trailing_scale must be > 0")

    guard_cfg = stage11["trade_count_guard"]
    if float(guard_cfg["bias_max_drop_pct"]) < 0:
        raise ValueError("evaluation.stage11.trade_count_guard.bias_max_drop_pct must be >= 0")
    if float(guard_cfg["confirm_max_drop_pct"]) < 0:
        raise ValueError("evaluation.stage11.trade_count_guard.confirm_max_drop_pct must be >= 0")
    float(guard_cfg["material_pf_improvement"])
    float(guard_cfg["material_exp_lcb_improvement"])
    evaluation["stage11"] = stage11

    stage11_55 = _merge_defaults(STAGE11_55_DEFAULTS, evaluation.get("stage11_55", {}))
    cache_cfg = stage11_55["cache"]
    if int(cache_cfg["max_entries_per_tf"]) < 1:
        raise ValueError("evaluation.stage11_55.cache.max_entries_per_tf must be >= 1")
    if float(cache_cfg["max_total_mb"]) <= 0:
        raise ValueError("evaluation.stage11_55.cache.max_total_mb must be > 0")
    evaluation["stage11_55"] = stage11_55

    stage12 = _merge_defaults(STAGE12_DEFAULTS, evaluation.get("stage12", {}))
    if not isinstance(stage12["enabled"], bool):
        raise ValueError("evaluation.stage12.enabled must be bool")
    if str(stage12.get("base_timeframe", "1m")).strip().lower() not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"evaluation.stage12.base_timeframe must be one of {SUPPORTED_TIMEFRAMES}")
    symbols = stage12.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("evaluation.stage12.symbols must be a non-empty list")
    if any(not str(symbol).strip() for symbol in symbols):
        raise ValueError("evaluation.stage12.symbols entries must be non-empty strings")
    timeframes = stage12.get("timeframes", [])
    if not isinstance(timeframes, list) or not timeframes:
        raise ValueError("evaluation.stage12.timeframes must be a non-empty list")
    for idx, timeframe in enumerate(timeframes):
        if str(timeframe).strip().lower() not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"evaluation.stage12.timeframes[{idx}] must be one of {SUPPORTED_TIMEFRAMES}")
    if not isinstance(stage12.get("include_stage06_baselines", True), bool):
        raise ValueError("evaluation.stage12.include_stage06_baselines must be bool")
    if not isinstance(stage12.get("include_stage10_families", True), bool):
        raise ValueError("evaluation.stage12.include_stage10_families must be bool")

    exits = stage12["exits"]
    variants = exits.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("evaluation.stage12.exits.variants must be a non-empty list")
    allowed_variants = {"fixed_atr", "structure_trailing", "time_based"}
    if any(str(item) not in allowed_variants for item in variants):
        raise ValueError("evaluation.stage12.exits.variants has unsupported value")
    if float(exits["time_based_stop_atr_multiple"]) <= 0:
        raise ValueError("evaluation.stage12.exits.time_based_stop_atr_multiple must be > 0")
    if float(exits["time_based_take_profit_atr_multiple"]) <= 0:
        raise ValueError("evaluation.stage12.exits.time_based_take_profit_atr_multiple must be > 0")

    cost_scenarios = stage12.get("cost_scenarios", {})
    if not isinstance(cost_scenarios, dict):
        raise ValueError("evaluation.stage12.cost_scenarios must be a mapping")
    for name in ("low", "realistic", "high"):
        if name not in cost_scenarios:
            raise ValueError("evaluation.stage12.cost_scenarios must include low/realistic/high")
    low_cfg = cost_scenarios["low"]
    high_cfg = cost_scenarios["high"]
    realistic_cfg = cost_scenarios["realistic"]
    for prefix, payload in (("low", low_cfg), ("high", high_cfg)):
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation.stage12.cost_scenarios.{prefix} must be mapping")
        for field in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps"):
            if float(payload[field]) < 0:
                raise ValueError(f"evaluation.stage12.cost_scenarios.{prefix}.{field} must be >= 0")
        if int(payload["delay_bars"]) < 0:
            raise ValueError(f"evaluation.stage12.cost_scenarios.{prefix}.delay_bars must be >= 0")
    if not isinstance(realistic_cfg, dict):
        raise ValueError("evaluation.stage12.cost_scenarios.realistic must be mapping")
    if not isinstance(realistic_cfg.get("use_config_default", True), bool):
        raise ValueError("evaluation.stage12.cost_scenarios.realistic.use_config_default must be bool")

    robustness = stage12.get("robustness", {})
    penalties = robustness.get("instability_penalty", {})
    if not isinstance(penalties, dict):
        raise ValueError("evaluation.stage12.robustness.instability_penalty must be mapping")
    for label in ("STABLE", "UNSTABLE", "INSUFFICIENT_DATA"):
        if float(penalties.get(label, 0.0)) < 0:
            raise ValueError(f"evaluation.stage12.robustness.instability_penalty.{label} must be >= 0")
    if float(robustness.get("cost_sensitivity_penalty_weight", 1.0)) < 0:
        raise ValueError("evaluation.stage12.robustness.cost_sensitivity_penalty_weight must be >= 0")
    if int(stage12.get("min_usable_windows_valid", 1)) < 1:
        raise ValueError("evaluation.stage12.min_usable_windows_valid must be >= 1")

    mc_cfg = stage12.get("monte_carlo", {})
    if not isinstance(mc_cfg.get("enabled", True), bool):
        raise ValueError("evaluation.stage12.monte_carlo.enabled must be bool")
    if not 0 < float(mc_cfg.get("top_pct", 0.2)) <= 1:
        raise ValueError("evaluation.stage12.monte_carlo.top_pct must be in (0,1]")
    if str(mc_cfg.get("bootstrap", "block")) not in {"iid", "block"}:
        raise ValueError("evaluation.stage12.monte_carlo.bootstrap must be iid or block")
    if int(mc_cfg.get("block_size_trades", 10)) < 1:
        raise ValueError("evaluation.stage12.monte_carlo.block_size_trades must be >= 1")
    if int(mc_cfg.get("n_paths", 5000)) < 100:
        raise ValueError("evaluation.stage12.monte_carlo.n_paths must be >= 100")
    if float(mc_cfg.get("initial_equity", 10000.0)) <= 0:
        raise ValueError("evaluation.stage12.monte_carlo.initial_equity must be > 0")
    if not 0 < float(mc_cfg.get("ruin_dd_threshold", 0.5)) < 1:
        raise ValueError("evaluation.stage12.monte_carlo.ruin_dd_threshold must be in (0,1)")

    forensic_cfg = stage12.get("forensics", {})
    if float(forensic_cfg.get("suspicious_backtest_ms_threshold", 5.0)) < 0:
        raise ValueError("evaluation.stage12.forensics.suspicious_backtest_ms_threshold must be >= 0")
    context_cfg = forensic_cfg.get("context_model", {})
    for key in (
        "regime_alignment_weight",
        "volatility_alignment_weight",
        "trend_strength_weight",
        "chop_penalty_weight",
        "separation_effect_size_threshold",
    ):
        if float(context_cfg.get(key, 0.0)) < 0:
            raise ValueError(f"evaluation.stage12.forensics.context_model.{key} must be >= 0")
    if int(context_cfg.get("min_samples", 30)) < 1:
        raise ValueError("evaluation.stage12.forensics.context_model.min_samples must be >= 1")
    evaluation["stage12"] = stage12

    stage12_3 = _merge_defaults(STAGE12_3_DEFAULTS, evaluation.get("stage12_3", {}))
    if not isinstance(stage12_3.get("enabled", False), bool):
        raise ValueError("evaluation.stage12_3.enabled must be bool")
    soft_weights = stage12_3.get("soft_weights", {})
    if not isinstance(soft_weights.get("enabled", True), bool):
        raise ValueError("evaluation.stage12_3.soft_weights.enabled must be bool")
    min_weight = float(soft_weights.get("min_weight", 0.25))
    if not 0 < min_weight <= 1:
        raise ValueError("evaluation.stage12_3.soft_weights.min_weight must be in (0,1]")
    regime_weight = float(soft_weights.get("regime_mismatch_weight", 0.5))
    vol_weight = float(soft_weights.get("vol_mismatch_weight", 0.5))
    if not 0 < regime_weight <= 1:
        raise ValueError("evaluation.stage12_3.soft_weights.regime_mismatch_weight must be in (0,1]")
    if not 0 < vol_weight <= 1:
        raise ValueError("evaluation.stage12_3.soft_weights.vol_mismatch_weight must be in (0,1]")

    usability_adaptive = stage12_3.get("usability_adaptive", {})
    if not isinstance(usability_adaptive.get("enabled", True), bool):
        raise ValueError("evaluation.stage12_3.usability_adaptive.enabled must be bool")
    min_floor = int(usability_adaptive.get("min_floor", 5))
    max_floor = int(usability_adaptive.get("max_floor", 80))
    alpha = float(usability_adaptive.get("alpha", 0.35))
    if min_floor < 1:
        raise ValueError("evaluation.stage12_3.usability_adaptive.min_floor must be >= 1")
    if max_floor < min_floor:
        raise ValueError("evaluation.stage12_3.usability_adaptive.max_floor must be >= min_floor")
    if alpha < 0:
        raise ValueError("evaluation.stage12_3.usability_adaptive.alpha must be >= 0")
    evaluation["stage12_3"] = stage12_3

    stage12_4 = _merge_defaults(STAGE12_4_DEFAULTS, evaluation.get("stage12_4", {}))
    if not isinstance(stage12_4.get("enabled", False), bool):
        raise ValueError("evaluation.stage12_4.enabled must be bool")
    threshold_grid = stage12_4.get("threshold_grid", [])
    if not isinstance(threshold_grid, list) or not threshold_grid:
        raise ValueError("evaluation.stage12_4.threshold_grid must be a non-empty list")
    for idx, value in enumerate(threshold_grid):
        threshold = float(value)
        if not 0 <= threshold <= 1:
            raise ValueError(f"evaluation.stage12_4.threshold_grid[{idx}] must be in [0,1]")
    weight_values = stage12_4.get("weight_values", [])
    if not isinstance(weight_values, list) or not weight_values:
        raise ValueError("evaluation.stage12_4.weight_values must be a non-empty list")
    for idx, value in enumerate(weight_values):
        if float(value) <= 0:
            raise ValueError(f"evaluation.stage12_4.weight_values[{idx}] must be > 0")
    target = stage12_4.get("trade_rate_target", {})
    tpm_min = float(target.get("tpm_min", 2.0))
    tpm_max = float(target.get("tpm_max", 40.0))
    if tpm_min < 0:
        raise ValueError("evaluation.stage12_4.trade_rate_target.tpm_min must be >= 0")
    if tpm_max <= 0:
        raise ValueError("evaluation.stage12_4.trade_rate_target.tpm_max must be > 0")
    if tpm_max < tpm_min:
        raise ValueError("evaluation.stage12_4.trade_rate_target.tpm_max must be >= tpm_min")
    cache_cfg = stage12_4.get("cache", {})
    if not isinstance(cache_cfg.get("enabled", True), bool):
        raise ValueError("evaluation.stage12_4.cache.enabled must be bool")
    evaluation["stage12_4"] = stage12_4

    stage13 = _merge_defaults(STAGE13_DEFAULTS, evaluation.get("stage13", {}))
    if not isinstance(stage13.get("enabled", False), bool):
        raise ValueError("evaluation.stage13.enabled must be bool")
    if int(stage13.get("seed", 42)) < 0:
        raise ValueError("evaluation.stage13.seed must be >= 0")
    families_cfg = stage13.get("families", {})
    enabled_families = families_cfg.get("enabled", [])
    if not isinstance(enabled_families, list) or not enabled_families:
        raise ValueError("evaluation.stage13.families.enabled must be a non-empty list")
    allowed_families = {"price", "volatility", "flow"}
    if any(str(name) not in allowed_families for name in enabled_families):
        raise ValueError("evaluation.stage13.families.enabled contains unsupported family")
    composer_cfg = stage13.get("composer", {})
    if str(composer_cfg.get("mode", "weighted_sum")) not in {"vote", "weighted_sum", "gated"}:
        raise ValueError("evaluation.stage13.composer.mode must be vote|weighted_sum|gated")
    weights_cfg = composer_cfg.get("weights", {})
    if not isinstance(weights_cfg, dict):
        raise ValueError("evaluation.stage13.composer.weights must be mapping")
    for family in allowed_families:
        if float(weights_cfg.get(family, 0.0)) < 0:
            raise ValueError("evaluation.stage13.composer.weights values must be >= 0")
    gated_cfg = composer_cfg.get("gated", {})
    if str(gated_cfg.get("gate_family", "volatility")) not in allowed_families:
        raise ValueError("evaluation.stage13.composer.gated.gate_family must be a supported family")
    if not 0 <= float(gated_cfg.get("gate_threshold", 0.2)) <= 1:
        raise ValueError("evaluation.stage13.composer.gated.gate_threshold must be in [0,1]")
    if not 0 <= float(gated_cfg.get("entry_threshold", 0.25)) <= 1:
        raise ValueError("evaluation.stage13.composer.gated.entry_threshold must be in [0,1]")
    gates_cfg = stage13.get("gates", {})
    if float(gates_cfg.get("zero_trade_pct_max", 40.0)) < 0:
        raise ValueError("evaluation.stage13.gates.zero_trade_pct_max must be >= 0")
    if not 0 <= float(gates_cfg.get("min_trade_count_ratio_vs_baseline", 0.6)) <= 1:
        raise ValueError("evaluation.stage13.gates.min_trade_count_ratio_vs_baseline must be in [0,1]")
    if float(gates_cfg.get("min_walkforward_executed_true_pct", 1.0)) < 0:
        raise ValueError("evaluation.stage13.gates.min_walkforward_executed_true_pct must be >= 0")
    if float(gates_cfg.get("min_mc_trigger_rate", 1.0)) < 0:
        raise ValueError("evaluation.stage13.gates.min_mc_trigger_rate must be >= 0")
    for family in ("price", "volatility", "flow"):
        params = stage13.get(family, {})
        if not isinstance(params, dict):
            raise ValueError(f"evaluation.stage13.{family} must be mapping")
        if float(params.get("entry_threshold", 0.3)) < 0 or float(params.get("entry_threshold", 0.3)) > 1:
            raise ValueError(f"evaluation.stage13.{family}.entry_threshold must be in [0,1]")
        sweep_grid = params.get("sweep_grid", {})
        if not isinstance(sweep_grid, dict):
            raise ValueError(f"evaluation.stage13.{family}.sweep_grid must be mapping")
        for key, values in sweep_grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"evaluation.stage13.{family}.sweep_grid.{key} must be a non-empty list")
    evaluation["stage13"] = stage13

    stage14 = _merge_defaults(STAGE14_DEFAULTS, evaluation.get("stage14", {}))
    if not isinstance(stage14.get("enabled", False), bool):
        raise ValueError("evaluation.stage14.enabled must be bool")
    if int(stage14.get("seed", 42)) < 0:
        raise ValueError("evaluation.stage14.seed must be >= 0")
    models = stage14.get("models", {}).get("allowed", [])
    if not isinstance(models, list) or not models:
        raise ValueError("evaluation.stage14.models.allowed must be non-empty list")
    if not set(str(name) for name in models).issubset({"logreg_l2", "ridge"}):
        raise ValueError("evaluation.stage14.models.allowed supports logreg_l2|ridge")
    max_features = int(stage14.get("max_features", 20))
    if max_features < 1 or max_features > 20:
        raise ValueError("evaluation.stage14.max_features must be in [1,20]")
    trade_bounds = stage14.get("trade_rate_bounds", {})
    min_tpm = float(trade_bounds.get("min_tpm", 5.0))
    max_tpm = float(trade_bounds.get("max_tpm", 80.0))
    if min_tpm < 0:
        raise ValueError("evaluation.stage14.trade_rate_bounds.min_tpm must be >= 0")
    if max_tpm <= 0 or max_tpm < min_tpm:
        raise ValueError("evaluation.stage14.trade_rate_bounds.max_tpm must be >= min_tpm and > 0")
    weighting_cfg = stage14.get("weighting", {})
    if not isinstance(weighting_cfg.get("enabled", True), bool):
        raise ValueError("evaluation.stage14.weighting.enabled must be bool")
    l2_grid = weighting_cfg.get("l2_grid", [])
    if not isinstance(l2_grid, list) or not l2_grid:
        raise ValueError("evaluation.stage14.weighting.l2_grid must be non-empty list")
    if any(float(v) <= 0 for v in l2_grid):
        raise ValueError("evaluation.stage14.weighting.l2_grid values must be > 0")
    if float(weighting_cfg.get("coef_clip", 3.0)) <= 0:
        raise ValueError("evaluation.stage14.weighting.coef_clip must be > 0")
    if float(weighting_cfg.get("drift_threshold", 0.4)) < 0:
        raise ValueError("evaluation.stage14.weighting.drift_threshold must be >= 0")
    thr_cfg = stage14.get("threshold_calibration", {})
    if not isinstance(thr_cfg.get("enabled", True), bool):
        raise ValueError("evaluation.stage14.threshold_calibration.enabled must be bool")
    for key in ("low_grid", "high_grid"):
        values = thr_cfg.get(key, [])
        if not isinstance(values, list) or not values:
            raise ValueError(f"evaluation.stage14.threshold_calibration.{key} must be non-empty list")
        if any(float(v) < 0 or float(v) > 1 for v in values):
            raise ValueError(f"evaluation.stage14.threshold_calibration.{key} values must be in [0,1]")
    nested = stage14.get("nested_walkforward", {})
    if not isinstance(nested.get("enabled", True), bool):
        raise ValueError("evaluation.stage14.nested_walkforward.enabled must be bool")
    if int(nested.get("folds", 3)) < 2:
        raise ValueError("evaluation.stage14.nested_walkforward.folds must be >= 2")
    meta_cfg = stage14.get("meta_family", {})
    if not isinstance(meta_cfg.get("enabled", False), bool):
        raise ValueError("evaluation.stage14.meta_family.enabled must be bool")
    if int(meta_cfg.get("min_families_required", 2)) < 2:
        raise ValueError("evaluation.stage14.meta_family.min_families_required must be >= 2")
    evaluation["stage14"] = stage14

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


def _timeframe_to_minutes(timeframe: str) -> int:
    text = str(timeframe).strip().lower()
    if text not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if text.endswith("m"):
        return int(text[:-1])
    if text.endswith("h"):
        return int(text[:-1]) * 60
    if text.endswith("d"):
        return int(text[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _validate_timeframe_multiple(base_timeframe: str, child_timeframe: str, field_name: str) -> None:
    base_minutes = _timeframe_to_minutes(base_timeframe)
    child_minutes = _timeframe_to_minutes(child_timeframe)
    if child_minutes < base_minutes:
        raise ValueError(f"{field_name} must be >= universe.base_timeframe")
    if child_minutes % base_minutes != 0:
        raise ValueError(f"{field_name} must be an integer multiple of universe.base_timeframe")


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
