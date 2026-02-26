"""Configuration loader, validator, and hasher."""

from __future__ import annotations

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

    evaluation = config["evaluation"]
    if not isinstance(evaluation["stage0_enabled"], bool):
        raise ValueError("evaluation.stage0_enabled must be bool")


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
