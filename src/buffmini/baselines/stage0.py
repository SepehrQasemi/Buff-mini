"""Baseline strategy definitions for Stage-0 / Stage-0.5 / Stage-0.6."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from buffmini.types import StrategySpec


def donchian_breakout() -> StrategySpec:
    """Donchian breakout baseline."""

    return StrategySpec(
        name="Donchian Breakout",
        entry_rules="Long when close > Donchian(period) high; short when close < Donchian(period) low.",
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
        entry_rules="Long when RSI(14) < long_entry; short when RSI(14) > short_entry.",
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "rsi_period": 14,
            "rsi_long_entry": 30,
            "rsi_short_entry": 70,
            "regime_gate": {"long": True, "short": True},
        },
    )


def trend_pullback() -> StrategySpec:
    """Trend pullback baseline."""

    return StrategySpec(
        name="Trend Pullback",
        entry_rules=(
            "Long when EMA_fast > EMA_slow, close > EMA_signal, and RSI(14) < long_entry; "
            "short when EMA_fast < EMA_slow, close < EMA_signal, and RSI(14) > short_entry."
        ),
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "ema_fast": 50,
            "ema_slow": 200,
            "signal_ema": 20,
            "rsi_period": 14,
            "rsi_long_entry": 40,
            "rsi_short_entry": 60,
            "regime_gate": {"long": False, "short": False},
        },
    )


def bollinger_mean_reversion() -> StrategySpec:
    """Bollinger mean-reversion baseline for Stage-0.6."""

    return StrategySpec(
        name="Bollinger Mean Reversion",
        entry_rules="Long when close < lower_band and RSI(14) < long_entry; short when close > upper_band and RSI(14) > short_entry.",
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "rsi_period": 14,
            "rsi_long_entry": 40,
            "rsi_short_entry": 60,
            "regime_gate": {"long": False, "short": False},
        },
    )


def range_breakout_with_ema_trend_filter() -> StrategySpec:
    """Range breakout with intrinsic EMA trend filter for Stage-0.6."""

    return StrategySpec(
        name="Range Breakout w/ EMA Trend Filter",
        entry_rules=(
            "Long when close > Donchian(period) high and EMA_fast > EMA_slow; "
            "short when close < Donchian(period) low and EMA_fast < EMA_slow."
        ),
        exit_rules="ATR stop loss, ATR take profit, or time stop from backtest engine.",
        parameters={
            "channel_period": 55,
            "ema_fast": 50,
            "ema_slow": 200,
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
    key = _normalize_name(strategy.name)
    p = strategy.parameters

    if key == "donchianbreakout":
        period = int(p.get("channel_period", 20))
        high_col = _require_column(frame, f"donchian_high_{period}")
        low_col = _require_column(frame, f"donchian_low_{period}")
        long_cond = frame["close"] > frame[high_col]
        short_cond = frame["close"] < frame[low_col]
    elif key == "rsimeanreversion":
        long_entry = float(p.get("rsi_long_entry", p.get("oversold", 30)))
        short_entry = float(p.get("rsi_short_entry", p.get("overbought", 70)))
        long_cond = frame["rsi_14"] < long_entry
        short_cond = frame["rsi_14"] > short_entry
    elif key == "trendpullback":
        ema_fast = int(p.get("ema_fast", 50))
        ema_slow = int(p.get("ema_slow", p.get("ema_long", 200)))
        signal_ema = int(p.get("signal_ema", p.get("ema_fast", 20)))
        rsi_long = float(p.get("rsi_long_entry", p.get("rsi_long_threshold", 40)))
        rsi_short = float(p.get("rsi_short_entry", p.get("rsi_short_threshold", 60)))

        fast_col = _require_column(frame, f"ema_{ema_fast}")
        slow_col = _require_column(frame, f"ema_{ema_slow}")
        signal_col = _require_column(frame, f"ema_{signal_ema}")

        long_cond = (frame[fast_col] > frame[slow_col]) & (frame["close"] > frame[signal_col]) & (frame["rsi_14"] < rsi_long)
        short_cond = (frame[fast_col] < frame[slow_col]) & (frame["close"] < frame[signal_col]) & (frame["rsi_14"] > rsi_short)
    elif key == "bollingermeanreversion":
        period = int(p.get("bollinger_period", p.get("bb_period", 20)))
        if period != 20:
            raise ValueError("Only bollinger_period=20 is supported in Stage-1")

        std_mult = float(p.get("bollinger_std", p.get("bb_std", 2.0)))
        rsi_long = float(p.get("rsi_long_entry", 40))
        rsi_short = float(p.get("rsi_short_entry", 60))

        mid_col = _require_column(frame, "bb_mid_20")
        std_col = _require_column(frame, "bb_std_20")
        upper = frame[mid_col] + std_mult * frame[std_col]
        lower = frame[mid_col] - std_mult * frame[std_col]

        long_cond = (frame["close"] < lower) & (frame["rsi_14"] < rsi_long)
        short_cond = (frame["close"] > upper) & (frame["rsi_14"] > rsi_short)
    elif key in {"rangebreakoutematrendfilter", "rangebreakoutwematrendfilter"}:
        period = int(p.get("channel_period", 55))
        ema_fast = int(p.get("ema_fast", p.get("ema_trend", 50)))
        ema_slow = int(p.get("ema_slow", p.get("ema_long", 200)))

        high_col = _require_column(frame, f"donchian_high_{period}")
        low_col = _require_column(frame, f"donchian_low_{period}")
        fast_col = _require_column(frame, f"ema_{ema_fast}")
        slow_col = _require_column(frame, f"ema_{ema_slow}")

        long_cond = (frame["close"] > frame[high_col]) & (frame[fast_col] > frame[slow_col])
        short_cond = (frame["close"] < frame[low_col]) & (frame[fast_col] < frame[slow_col])
    else:
        msg = f"Unknown strategy: {strategy.name}"
        raise ValueError(msg)

    return long_cond, short_cond


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _require_column(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise ValueError(f"Missing required feature column: {column}")
    return column
