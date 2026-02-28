"""Stage-10 package."""

from buffmini.stage10.regimes import (
    REGIME_FEATURE_COLUMNS,
    REGIME_LABELS,
    REGIME_SCORE_COLUMNS,
    compute_regime_scores,
    ensure_stage10_regime_features,
    regime_distribution,
)
from buffmini.stage10.signals import (
    DEFAULT_SIGNAL_PARAMS,
    MEAN_REV_FAMILIES,
    SIGNAL_FAMILIES,
    TREND_FAMILIES,
    generate_signal_family,
    signal_default_params,
    signal_family_type,
)

__all__ = [
    "REGIME_FEATURE_COLUMNS",
    "REGIME_LABELS",
    "REGIME_SCORE_COLUMNS",
    "compute_regime_scores",
    "ensure_stage10_regime_features",
    "regime_distribution",
    "DEFAULT_SIGNAL_PARAMS",
    "MEAN_REV_FAMILIES",
    "SIGNAL_FAMILIES",
    "TREND_FAMILIES",
    "generate_signal_family",
    "signal_default_params",
    "signal_family_type",
]
