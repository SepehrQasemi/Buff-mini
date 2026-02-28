"""Stage-10.2 entry signal library."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage10.regimes import REGIME_RANGE

TREND_FAMILIES: tuple[str, ...] = (
    "BreakoutRetest",
    "MA_SlopePullback",
    "VolCompressionBreakout",
)
MEAN_REV_FAMILIES: tuple[str, ...] = (
    "BollingerSnapBack",
    "ATR_DistanceRevert",
    "RangeFade",
)
SIGNAL_FAMILIES: tuple[str, ...] = TREND_FAMILIES + MEAN_REV_FAMILIES

DEFAULT_SIGNAL_PARAMS: dict[str, dict[str, float | int]] = {
    "BreakoutRetest": {"donchian_period": 20, "retest_atr_k": 0.8},
    "MA_SlopePullback": {"slope_min": 0.003, "pullback_atr_k": 1.2},
    "VolCompressionBreakout": {"donchian_period": 20, "compression_z": -0.8, "expansion_z": 0.6},
    "BollingerSnapBack": {"rsi_low": 35, "rsi_high": 65},
    "ATR_DistanceRevert": {"distance_k": 2.0},
    "RangeFade": {"donchian_period": 20, "edge_atr_k": 0.6},
}


def signal_family_type(family: str) -> str:
    name = str(family)
    if name in TREND_FAMILIES:
        return "trend"
    if name in MEAN_REV_FAMILIES:
        return "mean_reversion"
    raise ValueError(f"Unknown signal family: {family}")


def signal_default_params(family: str) -> dict[str, float | int]:
    if family not in DEFAULT_SIGNAL_PARAMS:
        raise ValueError(f"Unknown signal family: {family}")
    return dict(DEFAULT_SIGNAL_PARAMS[family])


def generate_signal_family(
    frame: pd.DataFrame,
    family: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return long/short entries and shifted signal for one Stage-10 family."""

    name = str(family)
    p = signal_default_params(name)
    if params:
        p.update(params)

    required = {"close", "atr_14", "ema_50", "rsi_14", "bb_mid_20", "bb_upper_20_2", "bb_lower_20_2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for Stage-10 signals: {sorted(missing)}")

    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    atr = pd.to_numeric(frame["atr_14"], errors="coerce").astype(float)
    ema_50 = pd.to_numeric(frame["ema_50"], errors="coerce").astype(float)
    rsi = pd.to_numeric(frame["rsi_14"], errors="coerce").astype(float)
    bb_mid = pd.to_numeric(frame["bb_mid_20"], errors="coerce").astype(float)
    bb_upper = pd.to_numeric(frame["bb_upper_20_2"], errors="coerce").astype(float)
    bb_lower = pd.to_numeric(frame["bb_lower_20_2"], errors="coerce").astype(float)
    slope = pd.to_numeric(frame.get("ema_slope_50", np.nan), errors="coerce").astype(float)
    bw_z = pd.to_numeric(frame.get("bb_bandwidth_z_120", np.nan), errors="coerce").astype(float)

    if name == "BreakoutRetest":
        period = int(p["donchian_period"])
        high = _require_column(frame, f"donchian_high_{period}")
        low = _require_column(frame, f"donchian_low_{period}")
        breakout_long = close > high
        breakout_short = close < low
        retest_k = float(p["retest_atr_k"])
        retest_long = (close - bb_mid).abs() <= (atr * retest_k)
        retest_short = (close - bb_mid).abs() <= (atr * retest_k)
        long_entry = breakout_long & retest_long & close.gt(close.shift(1))
        short_entry = breakout_short & retest_short & close.lt(close.shift(1))
        strength = _clip01((close - bb_mid).abs() / (atr.replace(0.0, np.nan) + 1e-12))
    elif name == "MA_SlopePullback":
        slope_min = float(p["slope_min"])
        pullback_k = float(p["pullback_atr_k"])
        long_depth = (ema_50 - close) / (atr.replace(0.0, np.nan) + 1e-12)
        short_depth = (close - ema_50) / (atr.replace(0.0, np.nan) + 1e-12)
        long_entry = (slope > slope_min) & (long_depth >= 0.0) & (long_depth <= pullback_k) & close.gt(close.shift(1))
        short_entry = (slope < -slope_min) & (short_depth >= 0.0) & (short_depth <= pullback_k) & close.lt(close.shift(1))
        strength = _clip01((slope.abs() / max(slope_min, 1e-6)) * 0.5 + _clip01((pullback_k - long_depth.abs()) / max(pullback_k, 1e-6)) * 0.5)
    elif name == "VolCompressionBreakout":
        period = int(p["donchian_period"])
        high = _require_column(frame, f"donchian_high_{period}")
        low = _require_column(frame, f"donchian_low_{period}")
        compression = (bw_z < float(p["compression_z"])).fillna(False).astype(bool)
        expansion = (bw_z > float(p["expansion_z"])).fillna(False).astype(bool)
        trigger = compression.shift(1, fill_value=False) & expansion
        long_entry = trigger & (close > high)
        short_entry = trigger & (close < low)
        strength = _clip01((bw_z - float(p["compression_z"])) / max(float(p["expansion_z"]) - float(p["compression_z"]), 1e-6))
    elif name == "BollingerSnapBack":
        rsi_low = float(p["rsi_low"])
        rsi_high = float(p["rsi_high"])
        long_entry = (close < bb_lower) & (rsi < rsi_low)
        short_entry = (close > bb_upper) & (rsi > rsi_high)
        distance = (bb_mid - close).abs() / (atr.replace(0.0, np.nan) + 1e-12)
        strength = _clip01(distance / 2.0)
    elif name == "ATR_DistanceRevert":
        distance_k = float(p["distance_k"])
        z = (close - ema_50) / (atr.replace(0.0, np.nan) + 1e-12)
        long_entry = z < -distance_k
        short_entry = z > distance_k
        strength = _clip01(z.abs() / max(distance_k, 1e-6))
    elif name == "RangeFade":
        period = int(p["donchian_period"])
        high = _require_column(frame, f"donchian_high_{period}")
        low = _require_column(frame, f"donchian_low_{period}")
        edge_k = float(p["edge_atr_k"])
        near_high = (high - close) <= atr * edge_k
        near_low = (close - low) <= atr * edge_k
        regime = frame.get("regime_label_stage10", pd.Series("", index=frame.index)).astype(str)
        in_range = regime.eq(REGIME_RANGE)
        long_entry = in_range & near_low
        short_entry = in_range & near_high
        strength = _clip01(((high - low) / (atr.replace(0.0, np.nan) + 1e-12)) / 6.0)
    else:
        raise ValueError(f"Unknown signal family: {name}")

    long_clean = long_entry.fillna(False).astype(bool)
    short_clean = short_entry.fillna(False).astype(bool)
    raw_signal = np.where(long_clean, 1, np.where(short_clean, -1, 0))
    signal = pd.Series(raw_signal, index=frame.index).shift(1).fillna(0).astype(int)
    strength_series = pd.Series(strength, index=frame.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    strength_series = _clip01(strength_series)

    return pd.DataFrame(
        {
            "long_entry": long_clean,
            "short_entry": short_clean,
            "signal_strength": strength_series,
            "signal": signal,
            "signal_family": name,
        }
    )


def _require_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _clip01(values: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(values, pd.Series):
        arr = np.clip(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float), 0.0, 1.0)
        return pd.Series(arr, index=values.index, dtype=float)
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, 0.0, 1.0)
    return pd.Series(clipped, dtype=float)
