"""Stage-10.4 soft regime-aware activation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage10.regimes import get_family_score
from buffmini.stage10.signals import signal_family_type

DEFAULT_ACTIVATION_CONFIG: dict[str, float] = {
    "multiplier_min": 0.9,
    "multiplier_max": 1.1,
    "trend_boost": 1.05,
    "range_boost": 1.03,
    "vol_cut": 0.95,
    "chop_cut": 0.93,
}

_ACTIVATION_ALIASES: dict[str, str] = {
    "m_min": "multiplier_min",
    "m_max": "multiplier_max",
    "expansion_cut": "vol_cut",
}


def activation_multiplier(
    regime_scores: dict[str, float] | pd.Series,
    regime_confidence: float,
    signal_family: str,
    settings: dict[str, Any] | None = None,
) -> float:
    """Compute score-only soft activation multiplier for one observation."""

    cfg = _normalized_activation_cfg(settings)

    family_type = signal_family_type(signal_family)
    conf = float(np.clip(float(regime_confidence), 0.0, 1.0))
    score_family = get_family_score(signal_family, regime_scores)
    score_vol_expansion = _safe_score(regime_scores, "score_vol_expansion")
    score_chop = _safe_score(regime_scores, "score_chop")

    m = 1.0
    if family_type == "trend":
        m *= 1.0 + (float(cfg["trend_boost"]) - 1.0) * score_family
    elif family_type == "mean_reversion":
        m *= 1.0 + (float(cfg["range_boost"]) - 1.0) * score_family
    m *= 1.0 - (1.0 - float(cfg["vol_cut"])) * score_vol_expansion
    m *= 1.0 - (1.0 - float(cfg["chop_cut"])) * score_chop

    smooth = 1.0 + (m - 1.0) * conf
    return float(np.clip(smooth, float(cfg["multiplier_min"]), float(cfg["multiplier_max"])))


def apply_soft_activation(
    signal_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    signal_family: str,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach activation multipliers without changing entry triggers."""

    if "signal" not in signal_frame.columns:
        raise ValueError("signal_frame must include signal column")
    if "signal_strength" not in signal_frame.columns:
        raise ValueError("signal_frame must include signal_strength column")
    if "regime_confidence_stage10" not in regime_frame.columns:
        raise ValueError("regime_frame must include regime_confidence_stage10")
    for score_col in ("score_trend", "score_range", "score_vol_expansion", "score_chop"):
        if score_col not in regime_frame.columns:
            raise ValueError(f"regime_frame must include {score_col}")

    aligned = signal_frame.copy()
    confidence = pd.to_numeric(regime_frame["regime_confidence_stage10"], errors="coerce").fillna(0.0)
    score_cols = regime_frame[["score_trend", "score_range", "score_vol_expansion", "score_chop"]]
    multipliers = [
        activation_multiplier(
            regime_scores=score_cols.iloc[idx],
            regime_confidence=float(confidence.iloc[idx]),
            signal_family=signal_family,
            settings=settings,
        )
        for idx in range(len(aligned))
    ]
    aligned["activation_multiplier"] = pd.Series(multipliers, index=aligned.index, dtype=float)
    aligned["effective_strength"] = (
        pd.to_numeric(aligned["signal_strength"], errors="coerce").fillna(0.0)
        * aligned["activation_multiplier"]
    )
    aligned["effective_strength"] = aligned["effective_strength"].clip(lower=0.0)
    return aligned


def _normalized_activation_cfg(settings: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = dict(DEFAULT_ACTIVATION_CONFIG)
    if settings:
        for key, value in settings.items():
            canonical = _ACTIVATION_ALIASES.get(str(key), str(key))
            if canonical in cfg:
                cfg[canonical] = float(value)
    return cfg


def _safe_score(scores: dict[str, float] | pd.Series, key: str) -> float:
    if isinstance(scores, pd.Series):
        value = scores.get(key, 0.0)
    else:
        value = scores.get(key, 0.0)
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    return float(np.clip(numeric, 0.0, 1.0))
