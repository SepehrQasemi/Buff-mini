"""Stage-37 activation/data/self-learning utilities."""

from .activation import (
    ActivationHuntConfig,
    calibrate_thresholds,
    compute_activation_metrics,
    compute_reject_chain_metrics,
    mode_settings,
)
from .self_learning import (
    LearningRegistryEntry,
    compute_family_exploration_weights,
    prune_features_by_contribution,
    select_elites_deterministic,
    upsert_learning_registry_entry,
)

__all__ = [
    "ActivationHuntConfig",
    "LearningRegistryEntry",
    "calibrate_thresholds",
    "compute_activation_metrics",
    "compute_family_exploration_weights",
    "compute_reject_chain_metrics",
    "mode_settings",
    "prune_features_by_contribution",
    "select_elites_deterministic",
    "upsert_learning_registry_entry",
]
