"""Execution reject taxonomy and deterministic aggregation for Stage-23."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXECUTION_REJECT_REASONS: tuple[str, ...] = (
    "SIZE_ZERO",
    "SIZE_TOO_SMALL",
    "STOP_TOO_CLOSE",
    "STOP_INVALID",
    "RR_INVALID",
    "SLIPPAGE_TOO_HIGH",
    "SPREAD_TOO_HIGH",
    "MARGIN_FAIL",
    "POSITION_CONFLICT",
    "DELAY_FAIL",
    "NO_FILL",
    "POLICY_CAP_HIT",
    "EXECUTION_INFEASIBLE_CAP",
    "UNKNOWN",
)

_KNOWN_REASON_SET = set(EXECUTION_REJECT_REASONS)


def normalize_reject_reason(reason: str | None) -> str:
    """Map unknown or empty labels to UNKNOWN in one place."""

    text = str(reason or "").strip().upper()
    return text if text in _KNOWN_REASON_SET else "UNKNOWN"


@dataclass
class RejectBreakdown:
    """Deterministic reject aggregation payload."""

    total_orders_attempted: int = 0
    total_orders_accepted: int = 0
    reject_reason_counts: dict[str, int] = field(default_factory=dict)

    def register_attempt(self, count: int = 1) -> None:
        self.total_orders_attempted += int(max(0, count))

    def register_accept(self, count: int = 1) -> None:
        self.total_orders_accepted += int(max(0, count))

    def register_reject(self, reason: str | None, count: int = 1) -> None:
        normalized = normalize_reject_reason(reason)
        self.reject_reason_counts[normalized] = int(self.reject_reason_counts.get(normalized, 0)) + int(max(0, count))

    def to_payload(self) -> dict[str, Any]:
        attempted = int(max(0, self.total_orders_attempted))
        accepted = int(max(0, self.total_orders_accepted))
        if accepted > attempted:
            accepted = attempted
        rejected = int(max(0, attempted - accepted))

        # Reconcile sparse reason counts to deterministic totals.
        reason_counts = {reason: int(self.reject_reason_counts.get(reason, 0)) for reason in EXECUTION_REJECT_REASONS}
        known_rejected = int(sum(reason_counts.values()))
        if known_rejected < rejected:
            reason_counts["UNKNOWN"] = int(reason_counts.get("UNKNOWN", 0)) + int(rejected - known_rejected)
        elif known_rejected > rejected:
            overflow = int(known_rejected - rejected)
            trimmed_unknown = max(0, int(reason_counts.get("UNKNOWN", 0)) - overflow)
            reason_counts["UNKNOWN"] = trimmed_unknown

        rates = {
            reason: (float(count) / float(attempted) if attempted > 0 else 0.0)
            for reason, count in sorted(reason_counts.items())
        }
        return {
            "total_orders_attempted": attempted,
            "total_orders_accepted": accepted,
            "total_orders_rejected": rejected,
            "reject_reason_counts": {reason: int(count) for reason, count in sorted(reason_counts.items())},
            "reject_reason_rate": rates,
        }
