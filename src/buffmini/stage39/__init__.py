"""Stage-39 upstream signal generation helpers."""

from .signal_generation import (
    LayeredCandidateOutput,
    build_layered_candidates,
    summarize_layered_candidates,
    widen_context_label,
)

__all__ = [
    "LayeredCandidateOutput",
    "build_layered_candidates",
    "summarize_layered_candidates",
    "widen_context_label",
]

