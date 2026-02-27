"""Stage-6 regime-aware dynamic leverage helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from buffmini.regime.classifier import REGIME_RANGE, REGIME_TREND, REGIME_VOL_EXPANSION


def compute_recent_drawdown(equity_history: list[float], lookback_bars: int) -> float:
    """Compute recent peak-to-valley drawdown on trailing equity history."""

    if not equity_history:
        return 0.0
    lookback = max(1, int(lookback_bars))
    values = np.asarray(equity_history[-lookback:], dtype=float)
    if values.size == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0.0, (peak - values) / peak, 0.0)
    drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(drawdown))


def _discrete_clip(raw_leverage: float, allowed_levels: list[float]) -> float:
    if not allowed_levels:
        return float(raw_leverage)
    levels = sorted({float(level) for level in allowed_levels if float(level) > 0})
    if not levels:
        return float(raw_leverage)
    below = [level for level in levels if level <= float(raw_leverage)]
    if below:
        return float(max(below))
    return float(min(levels))


def compute_dynamic_leverage(
    base_leverage: float,
    regime: str,
    dd_recent: float,
    config: dict[str, Any],
) -> dict[str, float | str]:
    """Compute conservative regime-aware leverage with deterministic clipping."""

    base = float(base_leverage)
    if base <= 0:
        raise ValueError("base_leverage must be > 0")

    dynamic_cfg = dict(config)
    multipliers = {
        REGIME_TREND: float(dynamic_cfg.get("trend_multiplier", 1.2)),
        REGIME_RANGE: float(dynamic_cfg.get("range_multiplier", 0.9)),
        REGIME_VOL_EXPANSION: float(dynamic_cfg.get("vol_expansion_multiplier", 0.7)),
    }
    regime_key = str(regime)
    multiplier = multipliers.get(regime_key, 1.0)
    leverage_raw = float(base * multiplier)

    # Conservative cap for stressed regime.
    if regime_key == REGIME_VOL_EXPANSION:
        leverage_raw = min(leverage_raw, base)

    dd_soft_threshold = float(dynamic_cfg.get("dd_soft_threshold", 0.08))
    dd_soft_multiplier = float(dynamic_cfg.get("dd_soft_multiplier", 0.8))
    if float(dd_recent) > dd_soft_threshold:
        leverage_raw *= dd_soft_multiplier

    max_leverage = float(dynamic_cfg.get("max_leverage", base))
    leverage_soft_clipped = min(float(leverage_raw), max_leverage)
    allowed_levels = [float(value) for value in dynamic_cfg.get("allowed_levels", [])]
    leverage_clipped = _discrete_clip(raw_leverage=leverage_soft_clipped, allowed_levels=allowed_levels)
    leverage_clipped = min(float(leverage_clipped), max_leverage)
    leverage_clipped = max(0.0, float(leverage_clipped))

    return {
        "regime": regime_key,
        "dd_recent": float(dd_recent),
        "leverage_raw": float(leverage_raw),
        "leverage_clipped": float(leverage_clipped),
    }

