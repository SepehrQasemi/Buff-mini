"""Evidence/provenance contracts for validation and promotion decisions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


ALLOWED_METRIC_SOURCE_TYPES: tuple[str, ...] = (
    "real_replay",
    "real_walkforward",
    "real_monte_carlo",
    "real_cross_perturbation",
    "real_transfer",
    "heuristic_filter",
    "proxy_only",
    "synthetic",
)

ALLOWED_STAGE_ROLES: tuple[str, ...] = (
    "heuristic_filter",
    "real_validation",
    "reporting_only",
    "orchestration_only",
)

DECISION_BLOCKED_SOURCE_TYPES: tuple[str, ...] = ("proxy_only", "synthetic")

REAL_DECISION_SOURCE_TYPES: tuple[str, ...] = (
    "real_replay",
    "real_walkforward",
    "real_monte_carlo",
    "real_cross_perturbation",
)

REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "run_id",
    "config_hash",
    "data_hash",
    "seed",
    "metric_name",
    "metric_value",
    "metric_source_type",
    "artifact_path",
    "stage_origin",
)


def build_metric_evidence(
    *,
    candidate_id: str,
    run_id: str,
    config_hash: str,
    data_hash: str,
    seed: int,
    metric_name: str,
    metric_value: float,
    metric_source_type: str,
    artifact_path: str,
    stage_origin: str,
    used_for_decision: bool = True,
    stage_role: str = "real_validation",
    notes: str = "",
) -> dict[str, Any]:
    """Build one metric evidence record using a strict schema."""

    record = {
        "candidate_id": str(candidate_id).strip(),
        "run_id": str(run_id).strip(),
        "config_hash": str(config_hash).strip(),
        "data_hash": str(data_hash).strip(),
        "seed": int(seed),
        "metric_name": str(metric_name).strip(),
        "metric_value": _finite(metric_value, default=0.0),
        "metric_source_type": str(metric_source_type).strip(),
        "artifact_path": str(artifact_path).strip(),
        "stage_origin": str(stage_origin).strip(),
        "used_for_decision": bool(used_for_decision),
        "stage_role": str(stage_role).strip(),
        "notes": str(notes),
    }
    errors = _validate_metric_evidence_errors(record=record)
    if errors:
        raise ValueError(f"Invalid metric evidence: {errors}")
    return record


def validate_metric_evidence(
    record: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate schema-level evidence integrity with optional artifact resolution."""

    row = dict(record or {})
    errors = _validate_metric_evidence_errors(record=row)
    artifact_exists = False
    artifact_resolved_path = ""
    artifact_raw = str(row.get("artifact_path", "")).strip()
    if artifact_raw:
        artifact_path = Path(artifact_raw)
        if repo_root is not None and not artifact_path.is_absolute():
            artifact_path = Path(repo_root) / artifact_path
        artifact_exists = bool(artifact_path.exists())
        artifact_resolved_path = str(artifact_path)
    return {
        "valid": bool(not errors),
        "errors": errors,
        "record": row,
        "artifact_exists": artifact_exists,
        "artifact_resolved_path": artifact_resolved_path,
    }


def _validate_metric_evidence_errors(record: dict[str, Any]) -> list[str]:
    """Internal schema validation returning raw error list."""

    errors: list[str] = []
    missing = [key for key in REQUIRED_EVIDENCE_FIELDS if key not in record]
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
        return errors

    for key in ("candidate_id", "run_id", "config_hash", "data_hash", "metric_name", "artifact_path", "stage_origin"):
        if not str(record.get(key, "")).strip():
            errors.append(f"empty_{key}")

    source_type = str(record.get("metric_source_type", "")).strip()
    if source_type not in set(ALLOWED_METRIC_SOURCE_TYPES):
        errors.append(f"invalid_metric_source_type:{source_type}")

    stage_role = str(record.get("stage_role", "real_validation")).strip()
    if stage_role not in set(ALLOWED_STAGE_ROLES):
        errors.append(f"invalid_stage_role:{stage_role}")

    try:
        _ = int(record.get("seed", 0))
    except Exception:
        errors.append("invalid_seed")

    metric_value = _finite(record.get("metric_value", 0.0), default=float("nan"))
    if not math.isfinite(metric_value):
        errors.append("non_finite_metric_value")

    return errors


def validate_metric_evidence_batch(
    records: list[dict[str, Any]],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a batch of evidence records and resolve artifact existence."""

    root = Path(repo_root) if repo_root is not None else None
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(records):
        row = dict(raw or {})
        row_errors = _validate_metric_evidence_errors(record=row)
        if row_errors:
            errors.extend([f"row_{idx}:{item}" for item in row_errors])
            continue
        artifact_path = Path(str(row["artifact_path"]))
        if root is not None and not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        row["artifact_exists"] = bool(artifact_path.exists())
        row["artifact_resolved_path"] = str(artifact_path)
        normalized.append(row)
    return {
        "valid": bool(not errors),
        "errors": errors,
        "records": normalized,
    }


def decision_evidence_guard(
    records: list[dict[str, Any]],
    *,
    required_real_sources: list[str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Check that decision-driving metrics are real and artifact-backed."""

    required_sources = [str(v) for v in (required_real_sources or list(REAL_DECISION_SOURCE_TYPES))]
    validated = validate_metric_evidence_batch(records, repo_root=repo_root)
    if not bool(validated["valid"]):
        return {
            "allowed": False,
            "reason": "invalid_evidence_schema",
            "errors": validated["errors"],
            "missing_real_sources": required_sources,
            "blocked_decision_metrics": [],
            "present_real_sources": [],
        }

    decision_records = [row for row in validated["records"] if bool(row.get("used_for_decision", True))]
    blocked_decision_metrics: list[str] = []
    blocked_decision_metric_details: list[str] = []
    present_real_sources: set[str] = set()
    missing_artifact_metrics: list[str] = []
    for row in decision_records:
        source = str(row.get("metric_source_type", ""))
        metric_name = str(row.get("metric_name", ""))
        if source in set(DECISION_BLOCKED_SOURCE_TYPES):
            blocked_decision_metrics.append(metric_name)
            blocked_decision_metric_details.append(f"{metric_name}:{source}")
        if source in set(required_sources):
            present_real_sources.add(source)
        if not bool(row.get("artifact_exists", False)):
            missing_artifact_metrics.append(metric_name)

    missing_sources = [source for source in required_sources if source not in present_real_sources]
    errors = list(validated["errors"])
    if blocked_decision_metric_details:
        errors.append(f"blocked_decision_sources:{','.join(sorted(blocked_decision_metric_details))}")
    if missing_sources:
        errors.append(f"missing_required_real_sources:{','.join(sorted(missing_sources))}")
    if missing_artifact_metrics:
        errors.append(f"missing_decision_artifacts:{','.join(sorted(set(missing_artifact_metrics)))}")
    return {
        "allowed": bool(not errors),
        "reason": "" if not errors else "decision_evidence_not_sufficient",
        "errors": errors,
        "missing_real_sources": missing_sources,
        "blocked_decision_metrics": sorted(set(blocked_decision_metrics)),
        "blocked_decision_metric_details": sorted(set(blocked_decision_metric_details)),
        "present_real_sources": sorted(present_real_sources),
        "decision_metric_count": int(len(decision_records)),
    }


def stage_role_from_source(metric_source_type: str) -> str:
    source = str(metric_source_type).strip()
    if source in {"real_replay", "real_walkforward", "real_monte_carlo", "real_cross_perturbation", "real_transfer"}:
        return "real_validation"
    if source in {"heuristic_filter"}:
        return "heuristic_filter"
    if source in {"proxy_only", "synthetic"}:
        return "reporting_only"
    return "orchestration_only"


def _finite(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)
