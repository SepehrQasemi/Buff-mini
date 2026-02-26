"""Tests for configuration loading and hashing."""

from pathlib import Path

from buffmini.config import compute_config_hash, load_config


def test_load_config_success() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    assert "universe" in config
    assert config["universe"]["timeframe"] == "1h"
    assert config["evaluation"]["stage0_enabled"] is True


def test_compute_config_hash_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")

    assert compute_config_hash(config) == compute_config_hash(config)
