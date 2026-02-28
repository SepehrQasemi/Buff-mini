"""Stage-10 package."""

from buffmini.stage10.regimes import (
    REGIME_FEATURE_COLUMNS,
    REGIME_LABELS,
    REGIME_SCORE_COLUMNS,
    compute_regime_scores,
    ensure_stage10_regime_features,
    regime_calibration_diagnostics,
    regime_distribution,
)
from buffmini.stage10.signals import (
    BREAKOUT_FAMILIES,
    DEFAULT_SIGNAL_PARAMS,
    MEAN_REV_FAMILIES,
    SIGNAL_FAMILIES,
    TREND_FAMILIES,
    generate_signal_family,
    signal_default_params,
    signal_family_type,
)
from buffmini.stage10.exits import (
    EXIT_MODE_ALIASES,
    apply_breakeven_after_1r,
    decide_exit_reason,
    normalize_exit_mode,
    partial_take_profit,
    should_regime_flip_exit,
    trailing_stop_path,
    update_trailing_stop,
)
from buffmini.stage10.activation import (
    DEFAULT_ACTIVATION_CONFIG,
    activation_multiplier,
    apply_soft_activation,
)

__all__ = [
    "REGIME_FEATURE_COLUMNS",
    "REGIME_LABELS",
    "REGIME_SCORE_COLUMNS",
    "compute_regime_scores",
    "ensure_stage10_regime_features",
    "regime_calibration_diagnostics",
    "regime_distribution",
    "BREAKOUT_FAMILIES",
    "DEFAULT_SIGNAL_PARAMS",
    "MEAN_REV_FAMILIES",
    "SIGNAL_FAMILIES",
    "TREND_FAMILIES",
    "generate_signal_family",
    "signal_default_params",
    "signal_family_type",
    "EXIT_MODE_ALIASES",
    "apply_breakeven_after_1r",
    "decide_exit_reason",
    "normalize_exit_mode",
    "partial_take_profit",
    "should_regime_flip_exit",
    "trailing_stop_path",
    "update_trailing_stop",
    "DEFAULT_ACTIVATION_CONFIG",
    "activation_multiplier",
    "apply_soft_activation",
]
