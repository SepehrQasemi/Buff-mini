"""Tests for configuration loading and hashing."""

from copy import deepcopy
from pathlib import Path

import pytest

from buffmini.config import compute_config_hash, load_config, validate_config


def test_load_config_success() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    assert "universe" in config
    assert config["universe"]["timeframe"] == "1h"
    assert config["evaluation"]["stage0_enabled"] is True
    assert config["evaluation"]["stage06"]["window_months"] == 36
    assert config["evaluation"]["stage06"]["end_mode"] == "latest"
    assert config["evaluation"]["stage1"]["split_mode"] == "60_20_20"
    assert config["evaluation"]["stage1"]["min_holdout_trades"] == 50
    assert config["evaluation"]["stage1"]["recent_weight"] == 2.0
    assert config["evaluation"]["stage1"]["target_trades_per_month_holdout"] == 8
    assert config["evaluation"]["stage1"]["low_signal_penalty_weight"] == 1.0
    assert config["evaluation"]["stage1"]["min_trades_per_month_floor"] == 2
    assert config["evaluation"]["stage1"]["min_validation_exposure_ratio"] == 0.01
    assert config["evaluation"]["stage1"]["min_validation_active_days"] == 10
    assert config["evaluation"]["stage1"]["allow_rare_if_high_expectancy"] is False
    assert config["evaluation"]["stage1"]["rare_expectancy_threshold"] == 3.0
    assert config["evaluation"]["stage1"]["rare_penalty_relief"] == 0.1
    thresholds = config["evaluation"]["stage1"]["result_thresholds"]
    assert thresholds["TierA"]["min_exp_lcb_holdout"] == 0
    assert thresholds["TierA"]["min_effective_edge"] == 0
    assert thresholds["TierA"]["min_trades_per_month_holdout"] == 5
    assert thresholds["TierA"]["min_pf_adj_holdout"] == 1.1
    assert thresholds["TierA"]["max_drawdown_holdout"] == 0.15
    assert thresholds["TierA"]["min_exposure_ratio"] == 0.02
    assert thresholds["TierB"]["min_exp_lcb_holdout"] == 0
    assert thresholds["TierB"]["min_effective_edge"] == 0
    assert thresholds["TierB"]["min_trades_per_month_holdout"] == 2
    assert thresholds["TierB"]["min_pf_adj_holdout"] == 1.05
    assert thresholds["TierB"]["max_drawdown_holdout"] == 0.20
    assert thresholds["TierB"]["min_exposure_ratio"] == 0.02
    assert thresholds["NearMiss"]["min_exp_lcb_holdout"] == -5
    assert config["evaluation"]["stage1"]["promotion_holdout_months"] == [3, 6, 9, 12]
    assert config["data"]["backend"] == "parquet"
    assert config["data"]["include_futures_extras"] is False
    assert config["data"]["futures_extras"]["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert config["data"]["futures_extras"]["timeframe"] == "1h"
    assert config["data"]["futures_extras"]["max_fill_gap_bars"] == 8
    assert config["data"]["futures_extras"]["funding"]["z_windows"] == [30, 90]
    assert config["data"]["futures_extras"]["funding"]["trend_window"] == 24
    assert config["data"]["futures_extras"]["funding"]["abs_pctl_window"] == 4320
    assert config["data"]["futures_extras"]["funding"]["extreme_pctl"] == 0.95
    assert config["data"]["futures_extras"]["open_interest"]["chg_windows"] == [1, 24]
    assert config["data"]["futures_extras"]["open_interest"]["z_window"] == 30
    assert config["data"]["futures_extras"]["open_interest"]["oi_to_volume_window"] == 24
    assert config["cost_model"]["mode"] == "simple"
    assert config["cost_model"]["round_trip_cost_pct"] == 0.1
    assert config["cost_model"]["v2"]["slippage_bps_base"] == 0.5
    assert config["cost_model"]["v2"]["slippage_bps_vol_mult"] == 2.0
    assert config["cost_model"]["v2"]["spread_bps"] == 0.5
    assert config["cost_model"]["v2"]["delay_bars"] == 0
    assert config["cost_model"]["v2"]["vol_proxy"] == "atr_pct"
    assert config["cost_model"]["v2"]["vol_lookback"] == 14
    assert config["cost_model"]["v2"]["max_total_bps_per_side"] == 10.0
    assert config["portfolio"]["walkforward"]["min_usable_windows"] == 3
    assert config["portfolio"]["walkforward"]["min_forward_trades"] == 10
    assert config["portfolio"]["walkforward"]["min_forward_exposure"] == 0.01
    assert config["portfolio"]["walkforward"]["pf_clip_max"] == 5.0
    assert config["portfolio"]["walkforward"]["stability_metric"] == "exp_lcb"
    selector = config["portfolio"]["leverage_selector"]
    assert selector["methods"] == ["equal", "vol"]
    assert selector["leverage_levels"] == [1, 2, 3, 5, 10, 15, 20, 25, 50]
    assert selector["bootstrap"] == "block"
    assert selector["block_size_trades"] == 10
    assert selector["n_paths"] == 20000
    assert selector["seed"] == 42
    assert selector["initial_equity"] == 10000
    assert selector["ruin_dd_threshold"] == 0.5
    assert selector["constraints"]["max_p_ruin"] == 0.01
    assert selector["constraints"]["max_dd_p95"] == 0.25
    assert selector["constraints"]["min_return_p05"] == 0.0
    assert selector["utility"]["objective"] == "expected_log_growth"
    assert selector["utility"]["epsilon"] == 1e-12
    assert config["execution"]["mode"] == "net"
    assert config["execution"]["per_symbol_netting"] is True
    assert config["risk"]["max_gross_exposure"] == 5.0
    assert config["risk"]["max_net_exposure_per_symbol"] == 5.0
    assert config["risk"]["sizing"]["mode"] == "risk_budget"
    assert config["risk"]["sizing"]["risk_per_trade_pct"] == 1.0
    assert config["risk"]["killswitch"]["enabled"] is True
    assert config["risk"]["killswitch"]["cool_down_bars"] == 48
    assert config["evaluation"]["stage4"]["default_method"] == "equal"
    assert config["evaluation"]["stage4"]["default_leverage"] == 1.0
    assert config["evaluation"]["stage6"]["enabled"] is False
    assert config["evaluation"]["stage6"]["regime"]["atr_percentile_window"] == 252
    assert config["evaluation"]["stage6"]["regime"]["vol_expansion_threshold"] == 0.80
    assert config["evaluation"]["stage6"]["regime"]["trend_strength_threshold"] == 0.010
    assert config["evaluation"]["stage6"]["confidence_sizing"]["scale"] == 2.0
    assert config["evaluation"]["stage6"]["dynamic_leverage"]["trend_multiplier"] == 1.2
    assert config["evaluation"]["stage6"]["dynamic_leverage"]["range_multiplier"] == 0.9
    assert config["evaluation"]["stage6"]["dynamic_leverage"]["vol_expansion_multiplier"] == 0.7
    assert config["evaluation"]["stage6"]["dynamic_leverage"]["dd_soft_threshold"] == 0.08
    assert config["evaluation"]["stage6"]["dynamic_leverage"]["max_leverage"] == 50.0
    assert config["evaluation"]["stage8"]["enabled"] is True
    assert config["evaluation"]["stage8"]["walkforward_v2"]["train_days"] == 180
    assert config["evaluation"]["stage8"]["walkforward_v2"]["holdout_days"] == 30
    assert config["evaluation"]["stage8"]["walkforward_v2"]["forward_days"] == 30
    assert config["evaluation"]["stage8"]["walkforward_v2"]["step_days"] == 30
    assert config["evaluation"]["stage8"]["walkforward_v2"]["min_trades"] == 10
    assert config["evaluation"]["stage8"]["walkforward_v2"]["min_exposure"] == 0.01
    assert config["ui"]["stage5"]["presets"]["quick"]["candidate_count"] == 1000
    assert config["ui"]["stage5"]["presets"]["full"]["candidate_count"] == 5000
    assert config["ui"]["stage5"]["window_months_options"] == [3, 6, 12, 36]


def test_validate_config_accepts_legacy_flat_result_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    legacy_config = deepcopy(config)
    legacy_config["evaluation"]["stage1"]["result_thresholds"] = {
        "min_exp_lcb_holdout": 0,
        "min_effective_edge": 0,
        "min_trades_per_month_holdout": 5,
        "min_pf_adj_holdout": 1.1,
        "max_drawdown_holdout": 0.15,
        "min_exposure_ratio": 0.02,
    }

    validate_config(legacy_config)

    thresholds = legacy_config["evaluation"]["stage1"]["result_thresholds"]
    assert thresholds["TierA"]["min_trades_per_month_holdout"] == 5
    assert thresholds["TierB"]["min_trades_per_month_holdout"] == 2
    assert thresholds["NearMiss"]["min_exp_lcb_holdout"] == -5


def test_compute_config_hash_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    assert compute_config_hash(config) == compute_config_hash(config)


def test_validate_config_rejects_unsorted_leverage_levels() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    config["portfolio"]["leverage_selector"]["leverage_levels"] = [1, 3, 2]

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_config(config)
