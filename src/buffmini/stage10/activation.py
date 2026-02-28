"""Stage-10.4 soft regime-aware activation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage10.regimes import REGIME_CHOP, REGIME_RANGE, REGIME_TREND, REGIME_VOL_EXPANSION
from buffmini.stage10.signals import signal_family_type

DEFAULT_ACTIVATION_CONFIG: dict[str, float] = {
    "m_min": 0.5,
    "m_max": 1.5,
    "trend_boost": 1.2,
    "range_boost": 1.1,
    "expansion_cut": 0.8,
    "chop_cut": 0.7,
}


def activation_multiplier(
    regime_label: str,
    regime_confidence: float,
    signal_family: str,
    settings: dict[str, Any] | None = None,
) -> float:
    """Compute soft activation multiplier for one signal/regime observation."""

    cfg = dict(DEFAULT_ACTIVATION_CONFIG)
    if settings:
        cfg.update({key: float(value) for key, value in settings.items() if key in cfg})

    family_type = signal_family_type(signal_family)
    label = str(regime_label)
    conf = float(np.clip(float(regime_confidence), 0.0, 1.0))

    m = 1.0
    if label == REGIME_TREND and family_type == "trend":
        m *= float(cfg["trend_boost"])
    if label == REGIME_RANGE and family_type == "mean_reversion":
        m *= float(cfg["range_boost"])
    if label == REGIME_VOL_EXPANSION:
        m *= float(cfg["expansion_cut"])
    if label == REGIME_CHOP:
        m *= float(cfg["chop_cut"])

    smooth = 1.0 + (m - 1.0) * conf
    return float(np.clip(smooth, float(cfg["m_min"]), float(cfg["m_max"])))


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
    if "regime_label_stage10" not in regime_frame.columns:
        raise ValueError("regime_frame must include regime_label_stage10")
    if "regime_confidence_stage10" not in regime_frame.columns:
        raise ValueError("regime_frame must include regime_confidence_stage10")

    aligned = signal_frame.copy()
    labels = regime_frame["regime_label_stage10"].astype(str)
    confidence = pd.to_numeric(regime_frame["regime_confidence_stage10"], errors="coerce").fillna(0.0)
    multipliers = [
        activation_multiplier(
            regime_label=labels.iloc[idx],
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
