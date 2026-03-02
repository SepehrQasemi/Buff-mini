from __future__ import annotations

from copy import deepcopy

import pytest

from buffmini.config import load_config, validate_config
from buffmini.constants import DEFAULT_CONFIG_PATH


def test_stage24_defaults_load() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    stage24 = cfg["evaluation"]["stage24"]
    constraints = cfg["evaluation"]["constraints"]
    assert stage24["enabled"] is False
    assert stage24["sizing"]["mode"] == "risk_pct"
    assert float(stage24["sizing"]["risk_ladder"]["r_min"]) == 0.02
    assert float(stage24["sizing"]["risk_ladder"]["r_max"]) == 0.20
    assert stage24["simulation"]["initial_equities"] == [100, 1000, 10000, 100000]
    assert constraints["mode"] in {"live", "research"}
    assert float(constraints["live"]["min_trade_notional"]) > 0.0
    assert float(constraints["research"]["min_trade_notional"]) >= 0.0


def test_stage24_invalid_ranges_rejected() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    bad = deepcopy(cfg)
    bad["evaluation"]["stage24"]["sizing"]["risk_ladder"]["r_min"] = 0.3
    bad["evaluation"]["stage24"]["sizing"]["risk_ladder"]["r_max"] = 0.2
    with pytest.raises(ValueError, match="risk_ladder"):
        validate_config(bad)


def test_stage24_initial_equities_must_be_positive() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    bad = deepcopy(cfg)
    bad["evaluation"]["stage24"]["simulation"]["initial_equities"] = [100, 0]
    with pytest.raises(ValueError, match="initial_equities\\[1\\] must be > 0"):
        validate_config(bad)


def test_constraints_mode_and_ranges_validation() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    bad_mode = deepcopy(cfg)
    bad_mode["evaluation"]["constraints"]["mode"] = "invalid"
    with pytest.raises(ValueError, match="evaluation\\.constraints\\.mode"):
        validate_config(bad_mode)

    bad_live = deepcopy(cfg)
    bad_live["evaluation"]["constraints"]["live"]["min_trade_notional"] = 0.0
    with pytest.raises(ValueError, match="constraints\\.live\\.min_trade_notional"):
        validate_config(bad_live)
