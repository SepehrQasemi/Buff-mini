"""Stage-4 execution primitives."""

from buffmini.execution.allocator import Order, generate_orders
from buffmini.execution.policy import ExecutionMode, Signal, TargetExposure, apply_execution_policy
from buffmini.execution.risk import PortfolioState, RiskDecision, compute_position_size, enforce_exposure_caps, killswitch_update_and_decide
from buffmini.execution.simulator import build_signals_from_stage2, run_stage4_simulation, simulate_execution

__all__ = [
    "ExecutionMode",
    "Signal",
    "TargetExposure",
    "apply_execution_policy",
    "PortfolioState",
    "RiskDecision",
    "compute_position_size",
    "enforce_exposure_caps",
    "killswitch_update_and_decide",
    "Order",
    "generate_orders",
    "simulate_execution",
    "build_signals_from_stage2",
    "run_stage4_simulation",
]

