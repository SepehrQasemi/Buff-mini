"""Typed Stage-24 sizing configuration structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SizingMode = Literal["risk_pct", "alloc_pct"]


@dataclass(frozen=True)
class RiskLadderParams:
    enabled: bool = True
    r_min: float = 0.02
    r_max: float = 0.20
    e_ref: float = 1000.0
    r_ref: float = 0.08
    k: float = 0.5


@dataclass(frozen=True)
class RiskClampParams:
    dd_soft: float = 0.10
    dd_hard: float = 0.20
    dd_soft_mult: float = 0.7
    dd_hard_mult: float = 0.4
    losing_streak_soft: int = 3
    losing_streak_hard: int = 5
    streak_soft_mult: float = 0.7
    streak_hard_mult: float = 0.4


@dataclass(frozen=True)
class Stage24SizingConfig:
    mode: SizingMode = "risk_pct"
    alloc_pct: float = 0.25
    risk_pct_user: float | None = None
    risk_ladder: RiskLadderParams = field(default_factory=RiskLadderParams)
    clamps: RiskClampParams = field(default_factory=RiskClampParams)


@dataclass(frozen=True)
class Stage24OrderConstraints:
    min_trade_notional: float = 10.0
    allow_size_bump_to_min_notional: bool = True
    max_notional_pct_of_equity: float = 1.0


@dataclass(frozen=True)
class Stage24SimulationConfig:
    initial_equities: tuple[float, ...] = (100.0, 1000.0, 10000.0, 100000.0)
    seed: int = 42
    use_stage3_selector_if_available: bool = True


@dataclass(frozen=True)
class Stage24Config:
    enabled: bool = False
    base_timeframe: str = "1m"
    operational_timeframe: str = "1h"
    sizing: Stage24SizingConfig = field(default_factory=Stage24SizingConfig)
    order_constraints: Stage24OrderConstraints = field(default_factory=Stage24OrderConstraints)
    simulation: Stage24SimulationConfig = field(default_factory=Stage24SimulationConfig)
