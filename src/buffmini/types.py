"""Shared types for Buff-mini."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

ConfigDict = dict[str, Any]
Side = Literal["long", "short"]


@dataclass(frozen=True)
class StrategySpec:
    """Defines a baseline strategy and its documented rules."""

    name: str
    entry_rules: str
    exit_rules: str
    parameters: dict[str, Any]


@dataclass
class Trade:
    """Single executed trade."""

    strategy: str
    symbol: str
    side: Side
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert trade to a serializable dictionary."""
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Backtest output bundle."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
