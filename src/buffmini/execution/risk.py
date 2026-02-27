"""Global Stage-4 risk controls: sizing, caps, and kill-switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PortfolioState:
    """Mutable portfolio state used by risk decision functions."""

    equity: float
    peak_equity: float
    day_start_equity: float
    consecutive_losses: int = 0
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    cooldown_remaining_bars: int = 0
    current_day: pd.Timestamp | None = None


@dataclass(frozen=True)
class RiskDecision:
    """Decision emitted by the risk engine for a given bar."""

    allow_new_trades: bool
    scaled_exposure_multiplier: float
    reasons: list[str]


def compute_position_size(
    equity: float,
    risk_cfg: dict[str, Any],
    stop_distance_pct: float | None,
) -> float:
    """Compute notional fraction of equity to allocate before leverage."""

    _ = float(equity)
    sizing = dict(risk_cfg.get("sizing", {}))
    mode = str(sizing.get("mode", "risk_budget"))
    if mode == "risk_budget":
        if stop_distance_pct is None or float(stop_distance_pct) <= 0:
            raise ValueError("risk_budget sizing requires stop_distance_pct > 0")
        risk_per_trade = float(sizing["risk_per_trade_pct"]) / 100.0
        return float(risk_per_trade / float(stop_distance_pct))
    if mode == "fixed_fraction":
        return float(float(sizing["fixed_fraction_pct"]) / 100.0)
    raise ValueError(f"Unsupported sizing mode: {mode}")


def enforce_exposure_caps(
    desired_exposures: list[dict[str, Any]],
    leverage: float,
    risk_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], float, list[str]]:
    """Scale desired exposures by a single multiplier so caps hold."""

    if not desired_exposures:
        return [], 1.0, []

    lev = float(leverage)
    if lev <= 0:
        raise ValueError("leverage must be > 0")

    gross = float(sum(abs(float(item["exposure_fraction"])) for item in desired_exposures) * lev)
    by_symbol: dict[str, float] = {}
    for item in desired_exposures:
        symbol = str(item["symbol"])
        by_symbol[symbol] = by_symbol.get(symbol, 0.0) + float(item["exposure_fraction"]) * lev

    max_gross = float(risk_cfg["max_gross_exposure"])
    max_net_per_symbol = float(risk_cfg["max_net_exposure_per_symbol"])
    multipliers = [1.0]
    reasons: list[str] = []
    if gross > max_gross and gross > 0:
        multipliers.append(max_gross / gross)
        reasons.append("max_gross_exposure")
    for symbol, signed in by_symbol.items():
        abs_value = abs(float(signed))
        if abs_value > max_net_per_symbol and abs_value > 0:
            multipliers.append(max_net_per_symbol / abs_value)
            reasons.append(f"max_net_exposure_per_symbol:{symbol}")

    scale = float(min(multipliers))
    scale = max(0.0, min(1.0, scale))
    if scale < 1.0 and "scaled_for_caps" not in reasons:
        reasons.append("scaled_for_caps")

    scaled: list[dict[str, Any]] = []
    for item in desired_exposures:
        copied = dict(item)
        copied["exposure_fraction"] = float(copied["exposure_fraction"]) * scale
        scaled.append(copied)
    return scaled, scale, reasons


def killswitch_update_and_decide(
    state: PortfolioState,
    pnl_change: float,
    ts: pd.Timestamp,
    bar_index: int,
    cfg: dict[str, Any],
) -> RiskDecision:
    """Update state and determine if new trades are allowed on this bar."""

    _ = int(bar_index)
    killswitch = dict(cfg.get("killswitch", {}))
    timestamp = pd.Timestamp(ts)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    current_day = timestamp.floor("D")
    if state.current_day is None or current_day != state.current_day:
        state.current_day = current_day
        state.day_start_equity = float(state.equity)

    state.equity = float(state.equity + float(pnl_change))
    state.peak_equity = float(max(float(state.peak_equity), float(state.equity)))

    pnl_value = float(pnl_change)
    if pnl_value < 0:
        state.consecutive_losses = int(state.consecutive_losses) + 1
    elif pnl_value > 0:
        state.consecutive_losses = 0

    if int(state.cooldown_remaining_bars) > 0:
        state.cooldown_remaining_bars = int(state.cooldown_remaining_bars) - 1
        return RiskDecision(False, 0.0, ["cooldown_active"])

    if not bool(killswitch.get("enabled", True)):
        return RiskDecision(True, 1.0, [])

    day_start = float(state.day_start_equity) if float(state.day_start_equity) != 0 else float(state.equity)
    peak = float(state.peak_equity) if float(state.peak_equity) != 0 else float(state.equity)
    daily_loss_pct = max(0.0, ((day_start - float(state.equity)) / day_start) * 100.0) if day_start > 0 else 0.0
    drawdown_pct = max(0.0, ((peak - float(state.equity)) / peak) * 100.0) if peak > 0 else 0.0

    reasons: list[str] = []
    if daily_loss_pct >= float(killswitch["max_daily_loss_pct"]):
        reasons.append("max_daily_loss_pct")
    if drawdown_pct >= float(killswitch["max_peak_to_valley_dd_pct"]):
        reasons.append("max_peak_to_valley_dd_pct")
    if int(state.consecutive_losses) >= int(killswitch["max_consecutive_losses"]):
        reasons.append("max_consecutive_losses")

    if reasons:
        state.cooldown_remaining_bars = int(killswitch["cool_down_bars"])
        reasons.append(f"cooldown:{int(killswitch['cool_down_bars'])}")
        return RiskDecision(False, 0.0, reasons)

    return RiskDecision(True, 1.0, [])

