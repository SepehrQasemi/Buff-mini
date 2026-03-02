"""Stage-23 execution hardening helpers."""

from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS, RejectBreakdown
from buffmini.stage23.eligibility import evaluate_eligibility

__all__ = [
    "EXECUTION_REJECT_REASONS",
    "RejectBreakdown",
    "build_adaptive_orders",
    "evaluate_eligibility",
]


def __getattr__(name: str):
    if name == "build_adaptive_orders":
        from buffmini.stage23.order_builder import build_adaptive_orders

        return build_adaptive_orders
    raise AttributeError(f"module 'buffmini.stage23' has no attribute {name!r}")
