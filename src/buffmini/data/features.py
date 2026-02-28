"""Feature calculation for Stage-0..Stage-9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.constants import DERIVED_DATA_DIR
from buffmini.data.derived_store import load_derived_parquet
from buffmini.data.features_futures import (
    build_all_futures_features,
    registered_futures_feature_columns,
    synthetic_futures_extras,
)
from buffmini.regime.classifier import attach_regime_columns
from buffmini.stage9.oi_overlay import (
    OI_DEPENDENT_COLUMNS,
    compute_oi_overlay_window,
    mask_oi_columns,
    overlay_metadata_dict,
)
from buffmini.stage10.regimes import compute_regime_scores

BASE_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi_14",
    "atr_14",
    "atr_14_sma_50",
    "bb_mid_20",
    "bb_std_20",
    "bb_upper_20_2",
    "bb_lower_20_2",
    "donchian_high_20",
    "donchian_low_20",
    "donchian_high_55",
    "donchian_low_55",
    "donchian_high_100",
    "donchian_low_100",
    "trend_strength",
    "atr_percentile_252",
    "regime",
    "bb_bandwidth_20",
    "bb_bandwidth_z_120",
    "atr_pct",
    "atr_pct_rank_252",
    "ema_slope_50",
    "trend_strength_stage10",
    "volume_z_120",
    "score_trend",
    "score_range",
    "score_vol_expansion",
    "score_vol_compression",
    "score_chop",
    "regime_label_stage10",
    "regime_confidence_stage10",
)


def registered_feature_columns(include_futures_extras: bool = False) -> list[str]:
    """Return canonical computed feature column names."""

    cols = list(CORE_FEATURE_COLUMNS)
    if bool(include_futures_extras):
        cols.extend(registered_futures_feature_columns())
    return cols


def calculate_features(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    symbol: str | None = None,
    timeframe: str = "1h",
    derived_data_dir: str | Path = DERIVED_DATA_DIR,
    _synthetic_extras_for_tests: bool = False,
) -> pd.DataFrame:
    """Calculate feature set without future leakage.

    When ``config.data.include_futures_extras`` is true, funding/open-interest
    features are merged from ``data/derived`` using timestamp alignment.
    """

    required = set(BASE_COLUMNS)
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
    data = compute_regime_scores(data)

    if _should_include_futures_extras(config):
        extras_cfg = dict(config.get("data", {}).get("futures_extras", {}))
        resolved_symbol = symbol or str(data.attrs.get("symbol") or "")
        if not resolved_symbol:
            raise ValueError("symbol is required when data.include_futures_extras=true")

        if _synthetic_extras_for_tests:
            funding_df, oi_df = synthetic_futures_extras(data)
        else:
            funding_df = load_derived_parquet(
                kind="funding",
                symbol=resolved_symbol,
                timeframe=timeframe,
                data_dir=derived_data_dir,
            )
            oi_df = load_derived_parquet(
                kind="open_interest",
                symbol=resolved_symbol,
                timeframe=timeframe,
                data_dir=derived_data_dir,
            )

        extras = build_all_futures_features(
            df_ohlcv=data,
            df_funding=funding_df,
            df_oi=oi_df,
            config={
                "timeframe": str(extras_cfg.get("timeframe", timeframe)),
                "max_fill_gap_bars": int(extras_cfg.get("max_fill_gap_bars", 8)),
                "funding": dict(extras_cfg.get("funding", {})),
                "open_interest": dict(extras_cfg.get("open_interest", {})),
            },
        )
        data = data.merge(extras, on="timestamp", how="left")

        if _oi_overlay_enabled(config):
            overlay_cfg = dict(
                (config.get("data", {}) or {})
                .get("futures_extras", {})
                .get("open_interest", {})
                .get("overlay", {})
            )
            recent_days = int(overlay_cfg.get("recent_window_days", 30))
            resolved_end_ts = (
                (config.get("universe", {}) or {}).get("resolved_end_ts")
                or (config.get("universe", {}) or {}).get("end")
            )
            window = compute_oi_overlay_window(
                df_ohlcv=data,
                df_oi=oi_df,
                resolved_end_ts=resolved_end_ts,
                recent_days=recent_days,
            )
            data, oi_active = mask_oi_columns(
                features_df=data,
                ts_col="timestamp",
                window_start_ts=window.window_start_ts,
                oi_columns=list(OI_DEPENDENT_COLUMNS),
            )
            data["oi_active"] = oi_active
            data.attrs["oi_overlay"] = overlay_metadata_dict(window=window, oi_active=oi_active, total_rows=len(data))

    return data


def _should_include_futures_extras(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        return False
    return bool(data_cfg.get("include_futures_extras", False))


def _oi_overlay_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        return False
    extras = data_cfg.get("futures_extras", {})
    if not isinstance(extras, dict):
        return False
    oi_cfg = extras.get("open_interest", {})
    if not isinstance(oi_cfg, dict):
        return False
    overlay_cfg = oi_cfg.get("overlay", {})
    if not isinstance(overlay_cfg, dict):
        return False
    return bool(overlay_cfg.get("enabled", False))
