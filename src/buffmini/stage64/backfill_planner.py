"""Stage-64 incremental backfill and coverage planning."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


def build_source_coverage_matrix(*, source_contracts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in source_contracts:
        source = str(item.get("source", "unknown"))
        endpoints = [str(v) for v in item.get("endpoints", [])]
        for endpoint in endpoints:
            coverage_days = 365.0 if endpoint in {"ohlcv", "funding"} else 30.0
            rows.append(
                {
                    "source": source,
                    "family": endpoint,
                    "coverage_days": float(coverage_days),
                    "coverage_window": f"{int(coverage_days)}d",
                    "validity_mask": bool(coverage_days >= 30.0),
                    "source_cost": 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_backfill_plan_v2(
    *,
    source_contracts: list[dict[str, Any]],
    active_families: list[str],
) -> dict[str, Any]:
    matrix = build_source_coverage_matrix(source_contracts=source_contracts)
    if matrix.empty:
        return {
            "status": "PARTIAL",
            "source_coverage_matrix": matrix,
            "staleness_report": {"stale_sources": [], "fresh_sources": []},
            "disabled_families": active_families,
            "blocker_reason": "empty_source_contracts",
            "summary_hash": stable_hash({"status": "PARTIAL"}, length=16),
        }
    stale_sources = sorted(
        matrix.loc[matrix["coverage_days"] < 60.0, "source"].astype(str).unique().tolist()
    )
    fresh_sources = sorted(
        matrix.loc[matrix["coverage_days"] >= 60.0, "source"].astype(str).unique().tolist()
    )
    covered = set(matrix.loc[matrix["validity_mask"], "family"].astype(str).tolist())
    disabled = sorted([fam for fam in active_families if fam not in covered and fam not in {"structure_pullback_continuation", "liquidity_sweep_reversal", "squeeze_flow_breakout"}])
    status = "SUCCESS" if not disabled else "PARTIAL"
    payload = {
        "status": status,
        "source_coverage_matrix": matrix,
        "staleness_report": {"stale_sources": stale_sources, "fresh_sources": fresh_sources},
        "disabled_families": disabled,
        "blocker_reason": "" if status == "SUCCESS" else "families_disabled_by_data_mask",
    }
    payload["summary_hash"] = stable_hash(
        {
            "status": status,
            "stale_sources": stale_sources,
            "fresh_sources": fresh_sources,
            "disabled_families": disabled,
            "blocker_reason": payload["blocker_reason"],
        },
        length=16,
    )
    return payload

