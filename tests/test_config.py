"""Tests for configuration loading and hashing."""

from pathlib import Path

from buffmini.config import compute_config_hash, load_config


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


def test_compute_config_hash_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    assert compute_config_hash(config) == compute_config_hash(config)
