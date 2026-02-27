"""Tests for configuration loading and hashing."""

from copy import deepcopy
from pathlib import Path

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
