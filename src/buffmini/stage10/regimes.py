"""Stage-10.1 regime scoring with leakage-safe derived features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REGIME_TREND = "TREND"
REGIME_RANGE = "RANGE"
REGIME_VOL_EXPANSION = "VOL_EXPANSION"
REGIME_VOL_COMPRESSION = "VOL_COMPRESSION"
REGIME_CHOP = "CHOP"

REGIME_LABELS: tuple[str, ...] = (
    REGIME_TREND,
    REGIME_RANGE,
    REGIME_VOL_EXPANSION,
    REGIME_VOL_COMPRESSION,
    REGIME_CHOP,
)

REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "bb_bandwidth_20",
    "bb_bandwidth_z_120",
    "atr_pct",
    "atr_pct_rank_252",
    "ema_slope_50",
    "trend_strength_stage10",
    "trend_strength_rank_252",
    "ema_slope_flip_rate_48",
    "volume_z_120",
)

REGIME_SCORE_COLUMNS: tuple[str, ...] = (
    "score_trend",
    "score_range",
    "score_vol_expansion",
    "score_vol_compression",
    "score_chop",
)

_TREND_FAMILIES: frozenset[str] = frozenset({"MA_SlopePullback"})
_RANGE_FAMILIES: frozenset[str] = frozenset({"BollingerSnapBack", "ATR_DistanceRevert", "RangeFade"})
_BREAKOUT_FAMILIES: frozenset[str] = frozenset({"BreakoutRetest", "VolCompressionBreakout"})


def ensure_stage10_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic derived columns required for Stage-10 regimes."""

    required = {"close", "volume", "ema_50", "atr_14", "bb_mid_20", "bb_upper_20_2", "bb_lower_20_2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for Stage-10 regimes: {sorted(missing)}")

    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    atr = pd.to_numeric(out["atr_14"], errors="coerce").astype(float)
    bb_mid = pd.to_numeric(out["bb_mid_20"], errors="coerce").astype(float)
    bb_upper = pd.to_numeric(out["bb_upper_20_2"], errors="coerce").astype(float)
    bb_lower = pd.to_numeric(out["bb_lower_20_2"], errors="coerce").astype(float)
    ema_50 = pd.to_numeric(out["ema_50"], errors="coerce").astype(float)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    bandwidth = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    out["bb_bandwidth_20"] = bandwidth
    out["bb_bandwidth_z_120"] = _rolling_zscore(bandwidth, window=120)
    out["atr_pct"] = atr / close.replace(0.0, np.nan)
    out["atr_pct_rank_252"] = _rolling_percentile(out["atr_pct"], window=252)
    out["ema_slope_50"] = (ema_50 / ema_50.shift(24)) - 1.0
    out["trend_strength_stage10"] = out["ema_slope_50"].abs()
    out["trend_strength_rank_252"] = _rolling_percentile(out["trend_strength_stage10"], window=252)
    slope_sign = np.sign(pd.to_numeric(out["ema_slope_50"], errors="coerce")).replace(0.0, np.nan).ffill().fillna(0.0)
    sign_flip = slope_sign.ne(slope_sign.shift(1)).astype(float)
    sign_flip.iloc[0] = 0.0
    out["ema_slope_flip_rate_48"] = sign_flip.rolling(window=48, min_periods=24).mean()
    out["volume_z_120"] = _rolling_zscore(volume, window=120)
    return out


def compute_regime_scores(
    frame: pd.DataFrame,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute per-bar regime probabilities and primary label."""

    cfg = dict(settings or {})
    trend_rank_strong = float(cfg.get("trend_rank_strong", 0.60))
    trend_rank_weak = float(cfg.get("trend_rank_weak", 0.40))
    vol_rank_high = float(cfg.get("high_vol_rank", cfg.get("vol_rank_high", 0.75)))
    vol_rank_low = float(cfg.get("low_vol_rank", cfg.get("vol_rank_low", 0.25)))
    chop_flip_window = int(cfg.get("chop_flip_window", 48))
    chop_flip_threshold = float(cfg.get("chop_flip_threshold", 0.18))
    compression_z = float(cfg.get("compression_z", -0.8))
    expansion_z = float(cfg.get("expansion_z", 1.0))
    volume_z_high = float(cfg.get("volume_z_high", 1.0))
    eps = 1e-12

    derived = ensure_stage10_regime_features(frame)
    out = derived.copy()

    trend_strength = pd.to_numeric(out["trend_strength_stage10"], errors="coerce")
    trend_rank = pd.to_numeric(out["trend_strength_rank_252"], errors="coerce")
    atr_rank = pd.to_numeric(out["atr_pct_rank_252"], errors="coerce")
    bb_z = pd.to_numeric(out["bb_bandwidth_z_120"], errors="coerce")
    volume_z = pd.to_numeric(out["volume_z_120"], errors="coerce")
    flip_rate = pd.to_numeric(out["ema_slope_flip_rate_48"], errors="coerce")
    if chop_flip_window != 48:
        slope_sign = np.sign(pd.to_numeric(out["ema_slope_50"], errors="coerce")).replace(0.0, np.nan).ffill().fillna(0.0)
        sign_flip = slope_sign.ne(slope_sign.shift(1)).astype(float)
        sign_flip.iloc[0] = 0.0
        flip_rate = sign_flip.rolling(window=max(2, chop_flip_window), min_periods=max(2, chop_flip_window // 2)).mean()
        out["ema_slope_flip_rate_48"] = flip_rate

    trend_scale = max(abs(trend_rank_strong - trend_rank_weak), 0.05)
    trend_score = _sigmoid((trend_rank - trend_rank_strong) / trend_scale)
    low_trend = 1.0 - trend_score

    mid_vol_score = 1.0 - np.abs(atr_rank - 0.5) / 0.25
    mid_vol_score = _clip01(mid_vol_score)
    range_vol_gate = _sigmoid((trend_rank_weak - trend_rank) / 0.08)
    range_atr_gate = _clip01(1.0 - np.abs(atr_rank - 0.45) / 0.30)
    range_bw_score = _clip01(1.0 - (bb_z.abs() / 2.0))
    range_score = range_vol_gate * range_atr_gate * range_bw_score

    vol_expansion_score = _clip01(
        (0.60 * _sigmoid((atr_rank - vol_rank_high) / 0.08))
        + (0.25 * _sigmoid((bb_z - expansion_z) * 1.5))
        + (0.15 * _sigmoid((volume_z - volume_z_high) * 1.2))
    )
    vol_compression_score = _sigmoid((vol_rank_low - atr_rank) / 0.08) * _sigmoid((compression_z - bb_z) * 1.5)

    chop_flip_score = _sigmoid((flip_rate - chop_flip_threshold) / 0.05)
    vol_neutral = _clip01(1.0 - np.abs(atr_rank - 0.5) / 0.25)
    chop_score = _clip01(((0.6 * low_trend) + (0.4 * chop_flip_score)) * vol_neutral)

    raw_scores = np.column_stack(
        [
            _nan_to_zero(trend_score),
            _nan_to_zero(range_score),
            _nan_to_zero(vol_expansion_score),
            _nan_to_zero(vol_compression_score),
            _nan_to_zero(chop_score),
        ]
    )
    denom = raw_scores.sum(axis=1, keepdims=True)
    normalized = np.divide(raw_scores, denom, out=np.zeros_like(raw_scores), where=denom > 0.0)

    for idx, column in enumerate(REGIME_SCORE_COLUMNS):
        out[column] = normalized[:, idx]

    labels = np.array(REGIME_LABELS)
    max_idx = np.argmax(normalized, axis=1)
    regime_label = labels[max_idx]
    out["regime_label_stage10"] = pd.Series(regime_label, index=out.index, dtype="object")
    out["regime_confidence_stage10"] = normalized.max(axis=1)
    return out


def regime_distribution(frame: pd.DataFrame) -> dict[str, float]:
    """Return deterministic regime percentage distribution."""

    if "regime_label_stage10" not in frame.columns:
        raise ValueError("regime_label_stage10 is required")
    series = frame["regime_label_stage10"].astype(str)
    counts = series.value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0) * 100.0) for label in REGIME_LABELS}


def regime_calibration_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return deterministic Stage-10.7 regime calibration diagnostics."""

    required = {"trend_strength_stage10", "atr_pct_rank_252", "regime_label_stage10"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for regime diagnostics: {sorted(missing)}")

    distribution = regime_distribution(frame)
    max_share = max(distribution.values()) if distribution else 0.0
    warnings: list[str] = []
    if max_share > 70.0:
        warnings.append("Single regime exceeds 70% share")
    atr_rank = pd.to_numeric(frame["atr_pct_rank_252"], errors="coerce").dropna()
    trend_strength = pd.to_numeric(frame["trend_strength_stage10"], errors="coerce").dropna()
    atr_stats = {
        "p05": float(atr_rank.quantile(0.05)) if not atr_rank.empty else 0.0,
        "p50": float(atr_rank.quantile(0.50)) if not atr_rank.empty else 0.0,
        "p95": float(atr_rank.quantile(0.95)) if not atr_rank.empty else 0.0,
    }
    return {
        "distribution_pct": distribution,
        "median_trend_strength": float(trend_strength.median()) if not trend_strength.empty else 0.0,
        "atr_percentile_distribution": atr_stats,
        "single_regime_warning": bool(max_share > 70.0),
        "warnings": warnings,
    }


def get_family_score(family: str, scores_row: dict[str, Any] | pd.Series) -> float:
    """Return score used by Stage-10.6 activation for the given signal family.

    Stage-10.6 uses score-only decisions:
    - trend families -> score_trend
    - mean-reversion families -> score_range
    - breakout families -> score_vol_expansion
    """

    row = dict(scores_row) if isinstance(scores_row, dict) else scores_row.to_dict()
    name = str(family)
    if name in _TREND_FAMILIES:
        value = row.get("score_trend", 0.0)
    elif name in _RANGE_FAMILIES:
        value = row.get("score_range", 0.0)
    elif name in _BREAKOUT_FAMILIES:
        value = row.get("score_vol_expansion", 0.0)
    else:
        value = max(
            float(row.get("score_trend", 0.0)),
            float(row.get("score_range", 0.0)),
            float(row.get("score_vol_expansion", 0.0)),
        )
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    return float(np.clip(numeric, 0.0, 1.0))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = s.rolling(window=window, min_periods=window).mean()
    sigma = s.rolling(window=window, min_periods=window).std(ddof=0)
    return (s - mu) / sigma.replace(0.0, np.nan)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)

    def _rank(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.isnan(arr).all():
            return np.nan
        last = arr[-1]
        return float(np.mean(arr <= last))

    return s.rolling(window=window, min_periods=window).apply(_rank, raw=True)


def _sigmoid(values: pd.Series | np.ndarray | float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-arr))


def _clip01(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _nan_to_zero(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
