"""Baseline strategy definitions for Stage-0."""

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
        parameters={"channel_period": 20},
    )


def rsi_mean_reversion() -> StrategySpec:
    """RSI mean-reversion baseline."""

    return StrategySpec(
        name="RSI Mean Reversion",
        entry_rules="Long when RSI(14) < 30; short when RSI(14) > 70.",
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={"rsi_period": 14, "oversold": 30, "overbought": 70},
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
        },
    )


def stage0_strategies() -> list[StrategySpec]:
    """Return the exact Stage-0 strategy set (3 strategies)."""

    return [donchian_breakout(), rsi_mean_reversion(), trend_pullback()]


def generate_signals(frame: pd.DataFrame, strategy: StrategySpec) -> pd.Series:
    """Generate -1/0/1 signal series for a Stage-0 strategy."""

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
    else:
        msg = f"Unknown strategy: {name}"
        raise ValueError(msg)

    raw_signal = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    # Shift by one bar so execution uses information from completed candles only.
    return pd.Series(raw_signal, index=frame.index).shift(1).fillna(0).astype(int)
