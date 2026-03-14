"""Stage-60 chain integrity checks for Stage-39..57."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.utils.hashing import stable_hash


_DOC_STAGE_SUMMARIES: tuple[str, ...] = (
    "stage39_signal_generation_summary.json",
    "stage47_signal_gen2_summary.json",
    "stage48_tradability_learning_summary.json",
    "stage52_summary.json",
    "stage53_summary.json",
    "stage54_summary.json",
    "stage55_summary.json",
    "stage56_summary.json",
    "stage57_summary.json",
)

_REQUIRED_RUN_ARTIFACTS: tuple[str, ...] = (
    "stage39/layer_a_candidates.csv",
    "stage47/setup_shortlist.csv",
    "stage48/stage48_ranked_candidates.csv",
    "stage48/stage48_stage_a_survivors.csv",
    "stage48/stage48_stage_b_survivors.csv",
    "stage52/setup_candidates_v2.csv",
    "stage53/predictions.csv",
)

_REQUIREMENT_STAGES: tuple[str, ...] = (
    "stage52",
    "stage53",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def detect_chain_id(*, stage28_run_id: str, docs_dir: Path) -> str:
    material = {
        "stage28_run_id": str(stage28_run_id).strip(),
        "summaries": {
            name: _load_json(docs_dir / name).get("summary_hash", "")
            for name in _DOC_STAGE_SUMMARIES
        },
    }
    return stable_hash(material, length=24)


def _extract_stage28_ids(docs_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _DOC_STAGE_SUMMARIES:
        payload = _load_json(docs_dir / name)
        stage = str(payload.get("stage", "")).strip()
        inferred = name.split("_")[0]
        key = stage or inferred
        out[str(key)] = str(payload.get("stage28_run_id", "")).strip()
    return out


def _required_artifacts_missing(*, runs_dir: Path, stage28_run_id: str) -> list[str]:
    if not str(stage28_run_id).strip():
        return [str(path) for path in _REQUIRED_RUN_ARTIFACTS]
    base = runs_dir / str(stage28_run_id).strip()
    missing: list[str] = []
    for rel in _REQUIRED_RUN_ARTIFACTS:
        if not (base / rel).exists():
            missing.append(str(rel))
    return missing


def assess_chain_integrity(
    *,
    docs_dir: Path,
    runs_dir: Path,
    budget_mode_selected: str,
) -> dict[str, Any]:
    stage_ids = _extract_stage28_ids(docs_dir)
    ids = [value for value in stage_ids.values() if value]
    stage28_run_id = ids[0] if ids else ""
    mismatched = {key: value for key, value in stage_ids.items() if value and value != stage28_run_id}

    missing_summaries = [name for name in _DOC_STAGE_SUMMARIES if not (docs_dir / name).exists()]
    missing_artifacts = _required_artifacts_missing(runs_dir=runs_dir, stage28_run_id=stage28_run_id)

    disallow_bootstrap = str(budget_mode_selected).lower() in {"validate", "full_audit"}
    bootstrap_stages: list[str] = []
    for stage_name in _REQUIREMENT_STAGES:
        payload = _load_json(docs_dir / f"{stage_name}_summary.json")
        mode = str(payload.get("input_mode", "")).strip().lower()
        if "bootstrap" in mode:
            bootstrap_stages.append(stage_name)

    blocker_parts: list[str] = []
    if missing_summaries:
        blocker_parts.append(f"missing_summaries:{','.join(sorted(missing_summaries))}")
    if mismatched:
        mismatch_txt = ",".join(f"{k}={v}" for k, v in sorted(mismatched.items(), key=lambda kv: kv[0]))
        blocker_parts.append(f"run_id_mismatch:{mismatch_txt}")
    if missing_artifacts:
        blocker_parts.append(f"missing_artifacts:{','.join(sorted(missing_artifacts))}")
    if disallow_bootstrap and bootstrap_stages:
        blocker_parts.append(f"bootstrap_forbidden:{','.join(sorted(bootstrap_stages))}")

    status = "SUCCESS" if not blocker_parts else "PARTIAL"
    chain_id = detect_chain_id(stage28_run_id=stage28_run_id, docs_dir=docs_dir)
    return {
        "stage": "60",
        "status": status,
        "stage28_run_id": stage28_run_id,
        "chain_id": chain_id,
        "budget_mode_selected": str(budget_mode_selected),
        "missing_summaries": sorted(missing_summaries),
        "run_id_mismatch": mismatched,
        "missing_artifacts": sorted(missing_artifacts),
        "bootstrap_forbidden": bool(disallow_bootstrap),
        "bootstrap_stages": sorted(bootstrap_stages),
        "blocker_reason": ";".join(blocker_parts),
        "summary_hash": stable_hash(
            {
                "status": status,
                "stage28_run_id": stage28_run_id,
                "chain_id": chain_id,
                "budget_mode_selected": str(budget_mode_selected),
                "missing_summaries": sorted(missing_summaries),
                "run_id_mismatch": mismatched,
                "missing_artifacts": sorted(missing_artifacts),
                "bootstrap_forbidden": bool(disallow_bootstrap),
                "bootstrap_stages": sorted(bootstrap_stages),
                "blocker_reason": ";".join(blocker_parts),
            },
            length=16,
        ),
    }

