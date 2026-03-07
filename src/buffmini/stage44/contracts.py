"""Stage-44 optimization framework contracts and validators."""

from __future__ import annotations

from typing import Any

ALLOWED_FAILURE_MOTIFS: tuple[str, ...] = (
    "NO_RAW_SIGNAL",
    "REJECT::NO_SIGNAL",
    "REJECT::BAD_RR",
    "REJECT::WEAK_FLOW_CONTEXT",
    "REJECT::FAILED_LIQUIDITY_CONFIRMATION",
    "REJECT::COST_DRAG",
    "REJECT::FEASIBILITY_FAIL",
    "REJECT::NO_STRUCTURE_CONFIRMATION",
)

REQUIRED_CONTRIBUTION_KEYS: tuple[str, ...] = (
    "module_name",
    "family_name",
    "setup_name",
    "raw_candidate_contribution",
    "stage_a_survival_lift",
    "stage_b_survival_lift",
    "final_policy_contribution",
    "runtime_seconds",
    "registry_rows_added",
    "cost_of_use_if_measurable",
    "coverage_flags",
)

REQUIRED_RUNTIME_KEYS: tuple[str, ...] = (
    "module_name",
    "phase_name",
    "enter_ts",
    "exit_ts",
    "elapsed_seconds",
    "candidate_rows_in",
    "candidate_rows_out",
)

REQUIRED_ALLOCATION_KEYS: tuple[str, ...] = (
    "module_name",
    "family_name",
    "exploration_eligible",
    "exploitation_score",
    "uncertainty_score",
    "novelty_score",
    "min_exploration_floor",
)

REQUIRED_REGISTRY_KEYS: tuple[str, ...] = (
    "module_name",
    "family_name",
    "setup_name",
    "contribution_summary",
    "failure_motifs",
    "runtime_metrics",
    "allocator_hook",
    "mutation_guidance",
)


def build_contribution_record(
    *,
    module_name: str,
    family_name: str,
    setup_name: str,
    raw_candidate_contribution: float,
    stage_a_survival_lift: float,
    stage_b_survival_lift: float,
    final_policy_contribution: float,
    runtime_seconds: float,
    registry_rows_added: int,
    cost_of_use_if_measurable: float | None,
    coverage_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    out = {
        "module_name": str(module_name),
        "family_name": str(family_name),
        "setup_name": str(setup_name),
        "raw_candidate_contribution": float(raw_candidate_contribution),
        "stage_a_survival_lift": float(stage_a_survival_lift),
        "stage_b_survival_lift": float(stage_b_survival_lift),
        "final_policy_contribution": float(final_policy_contribution),
        "runtime_seconds": float(runtime_seconds),
        "registry_rows_added": int(registry_rows_added),
        "cost_of_use_if_measurable": None if cost_of_use_if_measurable is None else float(cost_of_use_if_measurable),
        "coverage_flags": dict(coverage_flags or {}),
    }
    validate_contribution_record(out)
    return out


def validate_contribution_record(record: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_CONTRIBUTION_KEYS if key not in record]
    if missing:
        raise ValueError(f"Missing contribution keys: {missing}")
    for key in (
        "raw_candidate_contribution",
        "stage_a_survival_lift",
        "stage_b_survival_lift",
        "final_policy_contribution",
        "runtime_seconds",
    ):
        if not isinstance(record.get(key), (int, float)):
            raise ValueError(f"{key} must be numeric")
    if not isinstance(record.get("registry_rows_added"), int):
        raise ValueError("registry_rows_added must be int")
    if not isinstance(record.get("coverage_flags"), dict):
        raise ValueError("coverage_flags must be dict")


def build_failure_record(
    *,
    module_name: str,
    family_name: str,
    motif: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "module_name": str(module_name),
        "family_name": str(family_name),
        "motif": str(motif),
        "details": dict(details or {}),
    }
    validate_failure_record(out)
    return out


def validate_failure_record(record: dict[str, Any]) -> None:
    for key in ("module_name", "family_name", "motif", "details"):
        if key not in record:
            raise ValueError(f"{key} missing in failure record")
    motif = str(record.get("motif", ""))
    if motif not in set(ALLOWED_FAILURE_MOTIFS):
        raise ValueError(f"Unsupported failure motif: {motif}")
    if not isinstance(record.get("details"), dict):
        raise ValueError("details must be dict")


def build_runtime_event(
    *,
    module_name: str,
    phase_name: str,
    enter_ts: float,
    exit_ts: float,
    candidate_rows_in: int,
    candidate_rows_out: int,
) -> dict[str, Any]:
    elapsed = float(max(0.0, float(exit_ts) - float(enter_ts)))
    out = {
        "module_name": str(module_name),
        "phase_name": str(phase_name),
        "enter_ts": float(enter_ts),
        "exit_ts": float(exit_ts),
        "elapsed_seconds": float(elapsed),
        "candidate_rows_in": int(candidate_rows_in),
        "candidate_rows_out": int(candidate_rows_out),
    }
    validate_runtime_event(out)
    return out


def validate_runtime_event(event: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RUNTIME_KEYS if key not in event]
    if missing:
        raise ValueError(f"Missing runtime keys: {missing}")
    for key in ("enter_ts", "exit_ts", "elapsed_seconds"):
        if not isinstance(event.get(key), (int, float)):
            raise ValueError(f"{key} must be numeric")
    for key in ("candidate_rows_in", "candidate_rows_out"):
        if not isinstance(event.get(key), int):
            raise ValueError(f"{key} must be int")
    expected = float(max(0.0, float(event["exit_ts"]) - float(event["enter_ts"])))
    if abs(float(event["elapsed_seconds"]) - expected) > 1e-9:
        raise ValueError("elapsed_seconds must equal max(0, exit_ts-enter_ts)")


def build_allocator_hook(
    *,
    module_name: str,
    family_name: str,
    exploration_eligible: bool,
    exploitation_score: float,
    uncertainty_score: float,
    novelty_score: float,
    min_exploration_floor: float,
) -> dict[str, Any]:
    out = {
        "module_name": str(module_name),
        "family_name": str(family_name),
        "exploration_eligible": bool(exploration_eligible),
        "exploitation_score": float(exploitation_score),
        "uncertainty_score": float(uncertainty_score),
        "novelty_score": float(novelty_score),
        "min_exploration_floor": float(min_exploration_floor),
    }
    validate_allocator_hook(out)
    return out


def validate_allocator_hook(hook: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_ALLOCATION_KEYS if key not in hook]
    if missing:
        raise ValueError(f"Missing allocator keys: {missing}")
    if not isinstance(hook.get("exploration_eligible"), bool):
        raise ValueError("exploration_eligible must be bool")
    for key in ("exploitation_score", "uncertainty_score", "novelty_score", "min_exploration_floor"):
        if not isinstance(hook.get(key), (int, float)):
            raise ValueError(f"{key} must be numeric")


def to_registry_row(
    *,
    module_name: str,
    family_name: str,
    setup_name: str,
    contribution_summary: dict[str, Any],
    failure_motifs: list[str],
    runtime_metrics: dict[str, Any],
    allocator_hook: dict[str, Any],
    mutation_guidance: str,
) -> dict[str, Any]:
    out = {
        "module_name": str(module_name),
        "family_name": str(family_name),
        "setup_name": str(setup_name),
        "contribution_summary": dict(contribution_summary),
        "failure_motifs": [str(m) for m in failure_motifs],
        "runtime_metrics": dict(runtime_metrics),
        "allocator_hook": dict(allocator_hook),
        "mutation_guidance": str(mutation_guidance),
    }
    missing = [key for key in REQUIRED_REGISTRY_KEYS if key not in out]
    if missing:
        raise ValueError(f"Missing registry keys: {missing}")
    if not isinstance(out["failure_motifs"], list):
        raise ValueError("failure_motifs must be list")
    return out


def validate_stage44_summary(payload: dict[str, Any]) -> None:
    required = {
        "stage",
        "status",
        "contribution_contract_defined",
        "failure_contract_defined",
        "runtime_contract_defined",
        "allocator_hooks_defined",
        "registry_compatibility_defined",
        "modules_covered",
        "remaining_gaps",
        "summary_hash",
    }
    missing = sorted(required.difference(set(payload.keys())))
    if missing:
        raise ValueError(f"Missing Stage-44 summary keys: {missing}")
    if str(payload.get("stage", "")) != "44":
        raise ValueError("stage must be '44'")
    if not isinstance(payload.get("modules_covered"), list):
        raise ValueError("modules_covered must be list")
    if not isinstance(payload.get("remaining_gaps"), list):
        raise ValueError("remaining_gaps must be list")

