"""Stage-57 validation gates, stale input checks, and verdict logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from buffmini.validation import REAL_DECISION_SOURCE_TYPES, decision_evidence_guard


@dataclass(frozen=True)
class PromotionGates:
    replay_trade_count_min: int = 40
    replay_exp_lcb_min: float = 0.0
    replay_maxdd_max: float = 0.20
    replay_failure_dominance_max: float = 0.60
    walkforward_usable_windows_min: int = 5
    walkforward_median_exp_lcb_min: float = 0.0
    monte_carlo_downside_bound_min: float = 0.0
    cross_seed_survivors_min: int = 3
    required_real_sources: tuple[str, ...] = ()


def evaluate_replay_gate(metrics: dict[str, Any], *, gates: PromotionGates | None = None) -> dict[str, Any]:
    conf = gates or PromotionGates()
    trade_count = int(metrics.get("trade_count", 0))
    exp_lcb = float(metrics.get("exp_lcb", 0.0))
    maxdd = float(metrics.get("maxDD", metrics.get("max_drawdown", 0.0)))
    failure_dominance = float(metrics.get("failure_reason_dominance", 1.0))
    passed = (
        trade_count >= int(conf.replay_trade_count_min)
        and exp_lcb >= float(conf.replay_exp_lcb_min)
        and maxdd <= float(conf.replay_maxdd_max)
        and failure_dominance <= float(conf.replay_failure_dominance_max)
    )
    return {"passed": bool(passed), "trade_count": trade_count, "exp_lcb": exp_lcb, "maxDD": maxdd, "failure_reason_dominance": failure_dominance}


def evaluate_walkforward_gate(metrics: dict[str, Any], *, gates: PromotionGates | None = None) -> dict[str, Any]:
    conf = gates or PromotionGates()
    usable = int(metrics.get("usable_windows", 0))
    median_lcb = float(metrics.get("median_forward_exp_lcb", metrics.get("median_exp_lcb", 0.0)))
    passed = usable >= int(conf.walkforward_usable_windows_min) and median_lcb >= float(conf.walkforward_median_exp_lcb_min)
    return {"passed": bool(passed), "usable_windows": usable, "median_forward_exp_lcb": median_lcb}


def evaluate_monte_carlo_gate(metrics: dict[str, Any], *, gates: PromotionGates | None = None) -> dict[str, Any]:
    conf = gates or PromotionGates()
    downside = float(metrics.get("conservative_downside_bound", metrics.get("p05", -1.0)))
    passed = downside >= float(conf.monte_carlo_downside_bound_min)
    return {"passed": bool(passed), "conservative_downside_bound": downside}


def evaluate_cross_seed_gate(metrics: dict[str, Any], *, gates: PromotionGates | None = None) -> dict[str, Any]:
    conf = gates or PromotionGates()
    survivors = int(metrics.get("surviving_seeds", 0))
    passed = survivors >= int(conf.cross_seed_survivors_min)
    return {"passed": bool(passed), "surviving_seeds": survivors}


def validate_decision_evidence(
    evidence_records: list[dict[str, Any]] | None,
    *,
    required_real_sources: list[str] | tuple[str, ...] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    records = [dict(item) for item in (evidence_records or []) if isinstance(item, dict)]
    required = [str(v) for v in (required_real_sources or []) if str(v).strip()]
    if not records and not required:
        return {
            "allowed": True,
            "reason": "not_required",
            "errors": [],
            "missing_real_sources": [],
            "blocked_decision_metrics": [],
            "present_real_sources": [],
            "decision_metric_count": 0,
        }
    if not required:
        required = list(REAL_DECISION_SOURCE_TYPES)
    return decision_evidence_guard(records, required_real_sources=required, repo_root=repo_root)


def detect_stale_inputs(
    source_paths: list[Path | str],
    *,
    reference_time: datetime | None = None,
    max_age_hours: float = 24.0,
) -> dict[str, Any]:
    now = reference_time or datetime.now(timezone.utc)
    missing_paths: list[str] = []
    stale_paths: list[str] = []
    mtimes: dict[str, str] = {}
    cutoff = now - timedelta(hours=float(max_age_hours))
    for raw in source_paths:
        path = Path(raw)
        if not path.exists():
            missing_paths.append(str(path))
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        mtimes[str(path)] = mtime.isoformat()
        if mtime < cutoff:
            stale_paths.append(str(path))
    return {
        "stale": bool(missing_paths or stale_paths),
        "missing_paths": missing_paths,
        "stale_paths": stale_paths,
        "source_mtimes_utc": mtimes,
        "max_age_hours": float(max_age_hours),
    }


def derive_stage57_verdict(
    *,
    replay_metrics: dict[str, Any],
    walkforward_metrics: dict[str, Any],
    monte_carlo_metrics: dict[str, Any],
    cross_seed_metrics: dict[str, Any],
    evidence_records: list[dict[str, Any]] | None = None,
    validation_history: list[dict[str, Any]] | None = None,
    gates: PromotionGates | None = None,
) -> dict[str, Any]:
    conf = gates or PromotionGates()
    replay = evaluate_replay_gate(replay_metrics, gates=gates)
    walkforward = evaluate_walkforward_gate(walkforward_metrics, gates=gates)
    monte_carlo = evaluate_monte_carlo_gate(monte_carlo_metrics, gates=gates)
    cross_seed = evaluate_cross_seed_gate(cross_seed_metrics, gates=gates)
    evidence = validate_decision_evidence(
        evidence_records,
        required_real_sources=list(conf.required_real_sources),
        repo_root=Path("."),
    )
    all_gates_passed = bool(replay["passed"] and walkforward["passed"] and monte_carlo["passed"] and cross_seed["passed"])
    no_edge_in_scope = _scope_exhausted(validation_history or [])
    blocker_reason = ""
    if not bool(evidence.get("allowed", True)):
        verdict = "PARTIAL"
        blocker_reason = "decision_evidence_not_sufficient"
    elif all_gates_passed:
        verdict = "PASSING_EDGE"
    elif no_edge_in_scope:
        verdict = "NO_EDGE_IN_SCOPE"
    else:
        verdict = "PARTIAL"
    return {
        "verdict": verdict,
        "replay_gate": replay,
        "walkforward_gate": walkforward,
        "monte_carlo_gate": monte_carlo,
        "cross_seed_gate": cross_seed,
        "all_gates_passed": bool(all_gates_passed),
        "decision_evidence": evidence,
        "blocker_reason": blocker_reason,
    }


def _scope_exhausted(validation_history: list[dict[str, Any]]) -> bool:
    streak = 0
    for row in validation_history:
        verdict = str(row.get("verdict", "")).upper()
        scope_frozen = bool(row.get("scope_frozen", True))
        successful = verdict in {"PASSING_EDGE", "WEAK_EDGE", "MEDIUM_EDGE"}
        if scope_frozen and not successful:
            streak += 1
            if streak >= 3:
                return True
            continue
        streak = 0
    return False
