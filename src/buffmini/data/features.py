"""Feature calculation for Stage-0 and Stage-0.5 indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA/RSI/ATR/Donchian and ATR-SMA features without leakage."""

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

    # Shift Donchian channels by one bar to avoid using current candle breakout level.
    data["donchian_high_20"] = high.rolling(window=20, min_periods=20).max().shift(1)
    data["donchian_low_20"] = low.rolling(window=20, min_periods=20).min().shift(1)

    return data
