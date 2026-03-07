"""Stage-49 self-learning 3.0 helpers."""

from .self_learning3 import (
    deterministic_elites,
    expand_registry_rows_v3,
    failure_aware_mutation,
    family_module_downweighting,
    learning_depth_assessment,
)

__all__ = [
    "expand_registry_rows_v3",
    "failure_aware_mutation",
    "family_module_downweighting",
    "deterministic_elites",
    "learning_depth_assessment",
]

