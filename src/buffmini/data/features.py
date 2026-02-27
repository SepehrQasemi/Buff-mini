"""Feature calculation for Stage-0, Stage-0.5, Stage-0.6, and Stage-1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.regime.classifier import attach_regime_columns

def calculate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate feature set without future leakage."""

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    data = frame.copy()
    data = data.sort_values("timestamp").reset_index(drop=True)

    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)

    data["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    data["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    data["ema_100"] = close.ewm(span=100, adjust=False, min_periods=100).mean()
    data["ema_200"] = close.ewm(span=200, adjust=False, min_periods=200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    data["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()
    data["atr_14_sma_50"] = data["atr_14"].rolling(window=50, min_periods=50).mean()

    rolling_mid = close.rolling(window=20, min_periods=20)
    data["bb_mid_20"] = rolling_mid.mean()
    data["bb_std_20"] = rolling_mid.std(ddof=0)
    data["bb_upper_20_2"] = data["bb_mid_20"] + (2.0 * data["bb_std_20"])
    data["bb_lower_20_2"] = data["bb_mid_20"] - (2.0 * data["bb_std_20"])

    # Shift Donchian channels by one bar to avoid using current candle breakout level.
    data["donchian_high_20"] = high.rolling(window=20, min_periods=20).max().shift(1)
    data["donchian_low_20"] = low.rolling(window=20, min_periods=20).min().shift(1)
    data["donchian_high_55"] = high.rolling(window=55, min_periods=55).max().shift(1)
    data["donchian_low_55"] = low.rolling(window=55, min_periods=55).min().shift(1)
    data["donchian_high_100"] = high.rolling(window=100, min_periods=100).max().shift(1)
    data["donchian_low_100"] = low.rolling(window=100, min_periods=100).min().shift(1)

    data = attach_regime_columns(data)

    return data
