"""Stage-26 OHLCV-only context engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CONTEXTS = ["TREND", "RANGE", "VOL_EXPANSION", "VOL_COMPRESSION", "CHOP", "VOLUME_SHOCK"]


@dataclass(frozen=True)
class ContextParams:
    rank_window: int = 252
    vol_window: int = 24
    bb_window: int = 20
    volume_window: int = 120
    chop_window: int = 48
    trend_lookback: int = 24


def _rolling_rank(series: pd.Series, window: int) -> pd.Series:
    w = int(max(5, window))

    def _last_rank(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.5
        last = arr[-1]
        return float(np.mean(arr <= last))

    return pd.to_numeric(series, errors="coerce").rolling(window=w, min_periods=max(5, w // 8)).apply(_last_rank, raw=True).fillna(0.5)


def compute_context_features(df: pd.DataFrame, params: ContextParams | None = None) -> pd.DataFrame:
    """Compute deterministic causal context features from OHLCV only."""

    p = params or ContextParams()
    out = df.copy()
    close = pd.to_numeric(out.get("close", 0.0), errors="coerce").astype(float)
    high = pd.to_numeric(out.get("high", close), errors="coerce").astype(float)
    low = pd.to_numeric(out.get("low", close), errors="coerce").astype(float)
    volume = pd.to_numeric(out.get("volume", 0.0), errors="coerce").fillna(0.0).astype(float)

    if "atr_14" in out.columns:
        atr = pd.to_numeric(out["atr_14"], errors="coerce").replace(0.0, np.nan)
    else:
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=2).mean().replace(0.0, np.nan)

    ema_50 = pd.to_numeric(out.get("ema_50"), errors="coerce") if "ema_50" in out.columns else close.ewm(span=50, adjust=False).mean()
    ema_slope = ((ema_50 / ema_50.shift(max(1, int(p.trend_lookback)))) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trend_strength = ema_slope.abs()

    atr_pct = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = close.pct_change().rolling(int(max(5, p.vol_window)), min_periods=3).std(ddof=0).fillna(0.0) * np.sqrt(24.0)

    bb_mid = close.rolling(int(max(5, p.bb_window)), min_periods=3).mean()
    bb_std = close.rolling(int(max(5, p.bb_window)), min_periods=3).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = ((bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    direction = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))
    flips = (direction != np.roll(direction, 1)).astype(float)
    flips[0] = 0.0
    chop_score = pd.Series(flips, index=out.index).rolling(int(max(5, p.chop_window)), min_periods=4).mean().fillna(0.0)

    vol_mu = volume.rolling(int(max(10, p.volume_window)), min_periods=10).mean()
    vol_sigma = volume.rolling(int(max(10, p.volume_window)), min_periods=10).std(ddof=0).replace(0.0, np.nan)
    volume_z = ((volume - vol_mu) / vol_sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["ctx_atr_pct"] = atr_pct.astype(float)
    out["ctx_realized_vol"] = realized_vol.astype(float)
    out["ctx_bb_width"] = bb_width.astype(float)
    out["ctx_trend_strength"] = trend_strength.astype(float)
    out["ctx_chop_score"] = chop_score.astype(float)
    out["ctx_volume_z"] = volume_z.astype(float)
    out["ctx_ema_slope"] = ema_slope.astype(float)
    out["ctx_atr_pct_rank"] = _rolling_rank(out["ctx_atr_pct"], int(p.rank_window)).astype(float)
    out["ctx_bb_width_rank"] = _rolling_rank(out["ctx_bb_width"], int(p.rank_window)).astype(float)
    out["ctx_trend_rank"] = _rolling_rank(out["ctx_trend_strength"], int(p.rank_window)).astype(float)
    out["ctx_volume_z_rank"] = _rolling_rank(out["ctx_volume_z"], int(p.rank_window)).astype(float)
    return out


def classify_context(df: pd.DataFrame, params: ContextParams | None = None) -> pd.DataFrame:
    """Classify deterministic percentile-based contexts."""

    out = compute_context_features(df, params=params)
    trend_rank = pd.to_numeric(out["ctx_trend_rank"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    atr_rank = pd.to_numeric(out["ctx_atr_pct_rank"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    bbw_rank = pd.to_numeric(out["ctx_bb_width_rank"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    chop = pd.to_numeric(out["ctx_chop_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    volz_rank = pd.to_numeric(out["ctx_volume_z_rank"], errors="coerce").fillna(0.5).to_numpy(dtype=float)

    trend_score = np.clip((trend_rank - 0.40) / 0.60, 0.0, 1.0)
    range_mid_vol = 1.0 - np.clip(np.abs(atr_rank - 0.5) * 2.0, 0.0, 1.0)
    range_score = np.clip((0.7 * np.clip((0.40 - trend_rank) / 0.40, 0.0, 1.0)) + 0.3 * range_mid_vol, 0.0, 1.0)
    vol_exp_score = np.clip(np.maximum.reduce([atr_rank - 0.75, bbw_rank - 0.75, volz_rank - 0.80]) / 0.25, 0.0, 1.0)
    vol_cmp_score = np.clip(np.minimum(0.25 - atr_rank, 0.25 - bbw_rank) / 0.25, 0.0, 1.0)
    chop_mid_vol = 1.0 - np.clip(np.abs(atr_rank - 0.5) * 2.0, 0.0, 1.0)
    chop_score = np.clip(chop * 1.5 * np.clip((0.45 - trend_rank) / 0.45, 0.0, 1.0) * chop_mid_vol, 0.0, 1.0)
    volume_shock_score = np.clip((volz_rank - 0.90) / 0.10, 0.0, 1.0)

    score_mat = np.column_stack(
        [trend_score, range_score, vol_exp_score, vol_cmp_score, chop_score, volume_shock_score]
    )
    idx = np.argmax(score_mat, axis=1)
    labels = np.asarray(CONTEXTS, dtype=object)[idx]
    conf = np.max(score_mat, axis=1)

    out["ctx_score_trend"] = trend_score.astype(float)
    out["ctx_score_range"] = range_score.astype(float)
    out["ctx_score_vol_expansion"] = vol_exp_score.astype(float)
    out["ctx_score_vol_compression"] = vol_cmp_score.astype(float)
    out["ctx_score_chop"] = chop_score.astype(float)
    out["ctx_score_volume_shock"] = volume_shock_score.astype(float)
    out["ctx_state"] = pd.Series(labels, index=out.index, dtype="object")
    out["ctx_confidence"] = pd.Series(conf, index=out.index, dtype=float)
    return out

