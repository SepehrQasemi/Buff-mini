"""Stage-12 forensic helpers (execution and metric sanity diagnostics)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def classify_invalid_reason(
    *,
    trade_count: float,
    stability_classification: str,
    usable_windows: int,
    min_usable_windows_valid: int,
) -> str | None:
    """Return deterministic invalid reason label for one combination."""

    if float(trade_count) <= 0.0:
        return "ZERO_TRADE"
    if int(usable_windows) < int(min_usable_windows_valid):
        return "LOW_USABLE_WINDOWS"
    label = str(stability_classification)
    if label in {"INSUFFICIENT_DATA", "INVALID"}:
        return "INSUFFICIENT_DATA"
    return None


def metric_logic_validation(
    *,
    trade_count: float,
    profit_factor: float,
    pnl_values: np.ndarray,
    exp_lcb_reported: float,
    stability_classification: str,
    usable_windows: int,
    min_usable_windows_valid: int,
) -> dict[str, Any]:
    """Validate metric logic invariants without altering semantics."""

    pnl = np.asarray(pnl_values, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    recomputed = _exp_lcb(pnl)
    exp_lcb_ok = bool(abs(float(exp_lcb_reported) - float(recomputed)) <= 1e-12)
    zero_trade_pf_ok = bool(float(trade_count) > 0 or np.isfinite(float(profit_factor)))
    no_losing = bool(pnl.size > 0 and np.all(pnl >= 0.0))
    no_losing_pf_ok = bool((not no_losing) or np.isfinite(float(profit_factor)))
    stability_threshold_ok = not (
        int(usable_windows) < int(min_usable_windows_valid)
        and str(stability_classification) == "STABLE"
    )
    return {
        "exp_lcb_recomputed": float(recomputed),
        "exp_lcb_ok": bool(exp_lcb_ok),
        "zero_trade_pf_ok": bool(zero_trade_pf_ok),
        "no_losing_pf_ok": bool(no_losing_pf_ok),
        "stability_threshold_ok": bool(stability_threshold_ok),
        "all_ok": bool(exp_lcb_ok and zero_trade_pf_ok and no_losing_pf_ok and stability_threshold_ok),
    }


def aggregate_execution_diagnostics(
    matrix: pd.DataFrame,
    suspicious_backtest_ms_threshold: float = 5.0,
) -> dict[str, Any]:
    """Aggregate execution forensic matrix into summary diagnostics."""

    if matrix.empty:
        return {
            "total_combinations": 0,
            "avg_backtest_ms_per_combo": 0.0,
            "min_backtest_ms": 0.0,
            "max_backtest_ms": 0.0,
            "zero_trade_pct": 0.0,
            "invalid_pct": 0.0,
            "invalid_by_reason_pct": {},
            "walkforward_executed_true_pct": 0.0,
            "mc_trigger_rate": 0.0,
            "metric_logic_failures": 0,
            "walkforward_integrity_failures": 0,
            "suspicious_execution": True,
        }

    backtest_ms = pd.to_numeric(matrix["raw_backtest_seconds"], errors="coerce").fillna(0.0) * 1000.0
    total = float(len(matrix))
    invalid_mask = matrix["invalid_reason"].notna()
    zero_trade_mask = matrix["invalid_reason"].astype(str) == "ZERO_TRADE"
    walkforward_exec = matrix["walkforward_executed"].astype(bool)
    mc_exec = matrix["MC_executed"].astype(bool)
    logic_ok = matrix["metric_logic_all_ok"].astype(bool)
    wf_integrity_ok = matrix["walkforward_integrity_ok"].astype(bool)

    reason_counts = (
        matrix.loc[invalid_mask, "invalid_reason"]
        .astype(str)
        .value_counts(normalize=True)
        .sort_index()
    )
    invalid_by_reason_pct = {str(k): float(v * 100.0) for k, v in reason_counts.items()}

    avg_backtest_ms = float(backtest_ms.mean())
    summary = {
        "total_combinations": int(len(matrix)),
        "avg_backtest_ms_per_combo": avg_backtest_ms,
        "min_backtest_ms": float(backtest_ms.min()),
        "max_backtest_ms": float(backtest_ms.max()),
        "zero_trade_pct": float(zero_trade_mask.mean() * 100.0),
        "invalid_pct": float(invalid_mask.mean() * 100.0),
        "invalid_by_reason_pct": invalid_by_reason_pct,
        "walkforward_executed_true_pct": float(walkforward_exec.mean() * 100.0),
        "mc_trigger_rate": float(mc_exec.mean() * 100.0),
        "metric_logic_failures": int((~logic_ok).sum()),
        "walkforward_integrity_failures": int((~wf_integrity_ok).sum()),
        "suspicious_execution": bool(avg_backtest_ms < float(suspicious_backtest_ms_threshold)),
    }
    return summary


def classify_stage12_1(
    *,
    diagnostics: dict[str, Any],
    leaderboard: pd.DataFrame,
) -> str:
    """Classify Stage-12.1 forensic outcome."""

    if bool(diagnostics.get("suspicious_execution", False)) or int(diagnostics.get("walkforward_integrity_failures", 0)) > 0:
        return "ENGINE_BUG"
    if int(diagnostics.get("metric_logic_failures", 0)) > 0:
        return "METRIC_BUG"
    valid = leaderboard.loc[leaderboard["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return "TRUE_NO_EDGE"
    stable = valid.loc[valid["stability_classification"] == "STABLE"]
    if stable.empty and float(valid["exp_lcb"].max()) > 0:
        return "EDGE_PRESENT_BUT_INVALIDATED"
    return "TRUE_NO_EDGE"


def write_stage12_1_docs(
    *,
    report_md: Path,
    report_json: Path,
    run_id: str,
    diagnostics: dict[str, Any],
    classification: str,
) -> None:
    """Write Stage-12.1 execution forensic docs."""

    payload = dict(diagnostics)
    payload["run_id"] = str(run_id)
    payload["final_stage12_1_classification"] = str(classification)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-12.1 Execution Forensics & Sanity Audit")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- total_combinations: `{int(payload['total_combinations'])}`")
    lines.append(f"- avg_backtest_ms_per_combo: `{float(payload['avg_backtest_ms_per_combo']):.6f}`")
    lines.append(f"- min_backtest_ms: `{float(payload['min_backtest_ms']):.6f}`")
    lines.append(f"- max_backtest_ms: `{float(payload['max_backtest_ms']):.6f}`")
    lines.append(f"- zero_trade_pct: `{float(payload['zero_trade_pct']):.6f}`")
    lines.append(f"- invalid_pct: `{float(payload['invalid_pct']):.6f}`")
    lines.append(f"- walkforward_executed_true_pct: `{float(payload['walkforward_executed_true_pct']):.6f}`")
    lines.append(f"- mc_trigger_rate: `{float(payload['mc_trigger_rate']):.6f}`")
    lines.append(f"- suspicious_execution: `{bool(payload['suspicious_execution'])}`")
    lines.append(f"- metric_logic_failures: `{int(payload['metric_logic_failures'])}`")
    lines.append(f"- walkforward_integrity_failures: `{int(payload['walkforward_integrity_failures'])}`")
    lines.append("")
    lines.append("## Invalid Reason Distribution")
    lines.append("| reason | pct |")
    lines.append("| --- | ---: |")
    for reason, pct in sorted(payload.get("invalid_by_reason_pct", {}).items()):
        lines.append(f"| {reason} | {float(pct):.6f} |")
    lines.append("")
    lines.append("## Classification")
    lines.append(f"- `{classification}`")
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _exp_lcb(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean
    std = float(np.std(values, ddof=0))
    return float(mean - std / math.sqrt(float(values.size)))
