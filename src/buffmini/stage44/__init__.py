"""Stage-44 optimization framework contracts."""

from .contracts import (
    ALLOWED_FAILURE_MOTIFS,
    REQUIRED_ALLOCATION_KEYS,
    REQUIRED_CONTRIBUTION_KEYS,
    REQUIRED_REGISTRY_KEYS,
    REQUIRED_RUNTIME_KEYS,
    build_allocator_hook,
    build_contribution_record,
    build_failure_record,
    build_runtime_event,
    to_registry_row,
    validate_allocator_hook,
    validate_contribution_record,
    validate_failure_record,
    validate_runtime_event,
    validate_stage44_summary,
)

__all__ = [
    "ALLOWED_FAILURE_MOTIFS",
    "REQUIRED_CONTRIBUTION_KEYS",
    "REQUIRED_RUNTIME_KEYS",
    "REQUIRED_ALLOCATION_KEYS",
    "REQUIRED_REGISTRY_KEYS",
    "build_contribution_record",
    "validate_contribution_record",
    "build_failure_record",
    "validate_failure_record",
    "build_runtime_event",
    "validate_runtime_event",
    "build_allocator_hook",
    "validate_allocator_hook",
    "to_registry_row",
    "validate_stage44_summary",
]

