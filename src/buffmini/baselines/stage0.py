"""Baseline strategy definitions for Stage-0 / Stage-0.5 / Stage-0.6."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.types import StrategySpec


def donchian_breakout() -> StrategySpec:
    """Donchian breakout baseline."""

    return StrategySpec(
        name="Donchian Breakout",
        entry_rules="Long when close > Donchian(20) high; short when close < Donchian(20) low.",
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "channel_period": 20,
            "regime_gate": {"long": True, "short": True},
        },
    )


def rsi_mean_reversion() -> StrategySpec:
    """RSI mean-reversion baseline."""

    return StrategySpec(
        name="RSI Mean Reversion",
        entry_rules="Long when RSI(14) < 30; short when RSI(14) > 70.",
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "regime_gate": {"long": True, "short": True},
        },
    )


def trend_pullback() -> StrategySpec:
    """Trend pullback baseline."""

    return StrategySpec(
        name="Trend Pullback",
        entry_rules=(
            "Long when EMA50 > EMA200, close > EMA20, and RSI(14) < 40; "
            "short when EMA50 < EMA200, close < EMA20, and RSI(14) > 60."
        ),
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "ema_fast": 20,
            "ema_trend": 50,
            "ema_long": 200,
            "rsi_period": 14,
            "rsi_long_threshold": 40,
            "rsi_short_threshold": 60,
            "regime_gate": {"long": False, "short": False},
        },
    )


def bollinger_mean_reversion() -> StrategySpec:
    """Bollinger mean-reversion baseline for Stage-0.6."""

    return StrategySpec(
        name="Bollinger Mean Reversion",
        entry_rules="Long when close < lower_band_20_2 and RSI(14) < 40; short when close > upper_band_20_2 and RSI(14) > 60.",
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "regime_gate": {"long": False, "short": False},
        },
    )


def range_breakout_with_ema_trend_filter() -> StrategySpec:
    """Range breakout with intrinsic EMA trend filter for Stage-0.6."""

    return StrategySpec(
        name="Range Breakout w/ EMA Trend Filter",
        entry_rules=(
            "Long when close > Donchian(55) high and EMA50 > EMA200; "
            "short when close < Donchian(55) low and EMA50 < EMA200."
        ),
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "channel_period": 55,
            "ema_trend": 50,
            "ema_long": 200,
            "regime_gate": {"long": False, "short": False},
        },
    )


def stage0_strategies() -> list[StrategySpec]:
    """Return the exact Stage-0 strategy set (3 strategies)."""

    return [donchian_breakout(), rsi_mean_reversion(), trend_pullback()]


def stage06_strategies() -> list[StrategySpec]:
    """Return the Stage-0.6 expanded strategy set (5 strategies)."""

    return [
        donchian_breakout(),
        rsi_mean_reversion(),
        trend_pullback(),
        bollinger_mean_reversion(),
        range_breakout_with_ema_trend_filter(),
    ]


def generate_signals(
    frame: pd.DataFrame,
    strategy: StrategySpec,
    stage05: bool = False,
    gating_mode: str | None = None,
) -> pd.Series:
    """Generate -1/0/1 signal series for a strategy with optional gating.

    Gating modes:
    - ``none``: base strategy signals only
    - ``vol``: ATR volatility gate only
    - ``vol+regime``: ATR volatility gate + optional regime gate from strategy parameters

    For backward compatibility, ``stage05=True`` maps to ``vol+regime`` unless
    an explicit ``gating_mode`` is provided.
    """

    resolved_gating = gating_mode or ("vol+regime" if stage05 else "none")
    if resolved_gating not in {"none", "vol", "vol+regime"}:
        msg = f"Unknown gating mode: {resolved_gating}"
        raise ValueError(msg)

    long_cond, short_cond = _base_conditions(frame, strategy)

    if resolved_gating in {"vol", "vol+regime"}:
        required = {"atr_14", "atr_14_sma_50"}
        missing = required.difference(frame.columns)
        if missing:
            msg = f"Missing columns required for volatility gate: {sorted(missing)}"
            raise ValueError(msg)

        volatility_gate = frame["atr_14"] > frame["atr_14_sma_50"]
        long_cond = long_cond & volatility_gate
        short_cond = short_cond & volatility_gate

    if resolved_gating == "vol+regime":
        required = {"ema_50", "ema_200"}
        missing = required.difference(frame.columns)
        if missing:
            msg = f"Missing columns required for regime gate: {sorted(missing)}"
            raise ValueError(msg)

        regime_gate = strategy.parameters.get("regime_gate", {"long": False, "short": False})
        long_enabled = bool(regime_gate.get("long", False))
        short_enabled = bool(regime_gate.get("short", False))

        if long_enabled:
            long_cond = long_cond & (frame["ema_50"] > frame["ema_200"])
        if short_enabled:
            short_cond = short_cond & (frame["ema_50"] < frame["ema_200"])

    raw_signal = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    # Shift by one bar so execution uses information from completed candles only.
    return pd.Series(raw_signal, index=frame.index).shift(1).fillna(0).astype(int)


def _base_conditions(frame: pd.DataFrame, strategy: StrategySpec) -> tuple[pd.Series, pd.Series]:
    name = strategy.name

    if name == "Donchian Breakout":
        long_cond = frame["close"] > frame["donchian_high_20"]
        short_cond = frame["close"] < frame["donchian_low_20"]
    elif name == "RSI Mean Reversion":
        long_cond = frame["rsi_14"] < 30
        short_cond = frame["rsi_14"] > 70
    elif name == "Trend Pullback":
        long_cond = (frame["ema_50"] > frame["ema_200"]) & (frame["close"] > frame["ema_20"]) & (frame["rsi_14"] < 40)
        short_cond = (frame["ema_50"] < frame["ema_200"]) & (frame["close"] < frame["ema_20"]) & (frame["rsi_14"] > 60)
    elif name == "Bollinger Mean Reversion":
        long_cond = (frame["close"] < frame["bb_lower_20_2"]) & (frame["rsi_14"] < 40)
        short_cond = (frame["close"] > frame["bb_upper_20_2"]) & (frame["rsi_14"] > 60)
    elif name == "Range Breakout w/ EMA Trend Filter":
        long_cond = (frame["close"] > frame["donchian_high_55"]) & (frame["ema_50"] > frame["ema_200"])
        short_cond = (frame["close"] < frame["donchian_low_55"]) & (frame["ema_50"] < frame["ema_200"])
    else:
        msg = f"Unknown strategy: {name}"
        raise ValueError(msg)

    return long_cond, short_cond
