"""Stage-61 run-scoped writer for Stage-57 chain metrics with provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.validation import build_metric_evidence, decision_evidence_guard, stage_role_from_source
from buffmini.utils.hashing import stable_hash


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe(value: Any, *, default: float = 0.0) -> float:
    try:
        num = float(value)
    except Exception:
        return float(default)
    if pd.isna(num):  # type: ignore[arg-type]
        return float(default)
    return float(num)


def _replay_metrics_from_artifacts(*, stage48_dir: Path, stage53_dir: Path) -> tuple[dict[str, Any], str, Path]:
    real_path = stage53_dir / "replay_metrics_real.json"
    if real_path.exists():
        payload = _load_json(real_path)
        metrics = {
            "trade_count": int(max(0, _safe(payload.get("trade_count", 0.0)))),
            "exp_lcb": _safe(payload.get("exp_lcb", 0.0)),
            "maxDD": max(0.0, min(1.0, _safe(payload.get("maxDD", payload.get("max_drawdown", 1.0)), default=1.0))),
            "failure_reason_dominance": max(0.0, min(1.0, _safe(payload.get("failure_reason_dominance", 1.0), default=1.0))),
        }
        return metrics, "real_replay", real_path

    labels = _read_csv(stage48_dir / "stage48_labels.csv")
    survivors_b = _read_csv(stage53_dir / "stage_b_survivors.csv")
    trade_count = int(len(survivors_b))
    exp_lcb = _safe(pd.to_numeric(survivors_b.get("exp_lcb_proxy", 0.0), errors="coerce").fillna(0.0).mean()) if not survivors_b.empty else 0.0
    if not labels.empty:
        net = pd.to_numeric(labels.get("net_return_after_cost", 0.0), errors="coerce").fillna(0.0)
        dd_proxy = float(abs(min(0.0, float(net.min()))) / max(abs(float(net.max())), 1e-9)) if len(net) else 1.0
        fail_share = float((pd.to_numeric(labels.get("tradable", 0), errors="coerce").fillna(0).astype(int) == 0).mean())
    else:
        dd_proxy = 1.0
        fail_share = 1.0
    metrics = {
        "trade_count": int(trade_count),
        "exp_lcb": float(round(exp_lcb, 8)),
        "maxDD": float(round(min(1.0, max(0.0, dd_proxy)), 8)),
        "failure_reason_dominance": float(round(min(1.0, max(0.0, fail_share)), 8)),
    }
    return metrics, "heuristic_filter", stage48_dir / "stage48_labels.csv"


def _walkforward_metrics_from_artifacts(*, docs_dir: Path, runs_dir: Path, stage28_run_id: str) -> tuple[dict[str, Any], str, Path]:
    real_path = runs_dir / stage28_run_id / "stage67" / "walkforward_metrics_real.json"
    if real_path.exists():
        payload = _load_json(real_path)
        metrics = {
            "usable_windows": int(max(0, _safe(payload.get("usable_windows", 0.0)))),
            "median_forward_exp_lcb": _safe(payload.get("median_forward_exp_lcb", 0.0)),
        }
        return metrics, "real_walkforward", real_path

    stage67 = _load_json(docs_dir / "stage67_summary.json")
    metrics = {
        "usable_windows": int(max(0, _safe(stage67.get("split_count", 0.0)))),
        "median_forward_exp_lcb": _safe(stage67.get("mean_score", -0.01), default=-0.01),
    }
    source = "proxy_only" if stage67 else "synthetic"
    artifact = docs_dir / "stage67_summary.json"
    return metrics, source, artifact


def _monte_carlo_metrics_from_artifacts(*, docs_dir: Path, runs_dir: Path, stage28_run_id: str) -> tuple[dict[str, Any], str, Path]:
    real_path = runs_dir / stage28_run_id / "stage57" / "monte_carlo_metrics_real.json"
    if real_path.exists():
        payload = _load_json(real_path)
        metrics = {"conservative_downside_bound": _safe(payload.get("conservative_downside_bound", 0.0))}
        return metrics, "real_monte_carlo", real_path

    stage55 = _load_json(docs_dir / "stage55_summary.json")
    phase = dict(stage55.get("phase_timings", {}))
    mc_runtime = _safe(phase.get("monte_carlo", 0.0))
    metrics = {"conservative_downside_bound": float(-0.01 if mc_runtime <= 0.0 else -0.001)}
    source = "proxy_only" if stage55 else "synthetic"
    return metrics, source, docs_dir / "stage55_summary.json"


def _cross_perturbation_metrics_from_artifacts(*, docs_dir: Path, runs_dir: Path, stage28_run_id: str) -> tuple[dict[str, Any], str, Path]:
    real_path = runs_dir / stage28_run_id / "stage57" / "cross_perturbation_metrics_real.json"
    if real_path.exists():
        payload = _load_json(real_path)
        metrics = {"surviving_seeds": int(max(0, _safe(payload.get("surviving_seeds", 0.0))))}
        return metrics, "real_cross_perturbation", real_path

    seed_payload = _load_json(docs_dir / "stage50_5seed_summary.json")
    executed = [int(v) for v in seed_payload.get("executed_seeds", []) if str(v).strip().isdigit()]
    # Cross-seed is explicitly not treated as real perturbation robustness.
    metrics = {"surviving_seeds": int(min(5, len(executed))) if executed else 0}
    source = "proxy_only" if seed_payload else "synthetic"
    return metrics, source, docs_dir / "stage50_5seed_summary.json"


def _evidence_records_for_gate(
    *,
    metric_payload: dict[str, Any],
    metric_source_type: str,
    artifact_path: Path,
    candidate_id: str,
    stage28_run_id: str,
    config_hash: str,
    data_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metric_name, metric_value in sorted(metric_payload.items(), key=lambda kv: str(kv[0])):
        records.append(
            build_metric_evidence(
                candidate_id=candidate_id,
                run_id=stage28_run_id,
                config_hash=config_hash,
                data_hash=data_hash,
                seed=int(seed),
                metric_name=str(metric_name),
                metric_value=_safe(metric_value, default=0.0),
                metric_source_type=str(metric_source_type),
                artifact_path=str(artifact_path),
                stage_origin="stage61",
                used_for_decision=True,
                stage_role=stage_role_from_source(metric_source_type),
                notes="stage61_chain_metrics",
            )
        )
    return records


def materialize_stage57_chain_metrics(
    *,
    docs_dir: Path,
    runs_dir: Path,
    stage28_run_id: str,
    chain_id: str,
    config_hash: str,
    data_hash: str,
    seed: int,
    required_real_sources: list[str] | None = None,
) -> dict[str, Any]:
    if not str(stage28_run_id).strip():
        return {
            "status": "PARTIAL",
            "blocker_reason": "missing_stage28_run_id",
            "chain_metrics": {},
        }

    base = runs_dir / stage28_run_id
    stage48_dir = base / "stage48"
    stage53_dir = base / "stage53"
    required_dirs = [stage48_dir, stage53_dir]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        return {
            "status": "PARTIAL",
            "blocker_reason": f"missing_run_dirs:{','.join(sorted(missing))}",
            "chain_metrics": {},
        }

    replay, replay_source, replay_artifact = _replay_metrics_from_artifacts(stage48_dir=stage48_dir, stage53_dir=stage53_dir)
    walkforward, wf_source, wf_artifact = _walkforward_metrics_from_artifacts(
        docs_dir=docs_dir,
        runs_dir=runs_dir,
        stage28_run_id=stage28_run_id,
    )
    monte_carlo, mc_source, mc_artifact = _monte_carlo_metrics_from_artifacts(
        docs_dir=docs_dir,
        runs_dir=runs_dir,
        stage28_run_id=stage28_run_id,
    )
    cross_seed, cross_source, cross_artifact = _cross_perturbation_metrics_from_artifacts(
        docs_dir=docs_dir,
        runs_dir=runs_dir,
        stage28_run_id=stage28_run_id,
    )

    candidate_id = str(_load_json(docs_dir / "stage52_summary.json").get("representative_candidate_id", "__chain__"))
    evidence_records = (
        _evidence_records_for_gate(
            metric_payload=replay,
            metric_source_type=replay_source,
            artifact_path=replay_artifact,
            candidate_id=candidate_id,
            stage28_run_id=stage28_run_id,
            config_hash=config_hash,
            data_hash=data_hash,
            seed=seed,
        )
        + _evidence_records_for_gate(
            metric_payload=walkforward,
            metric_source_type=wf_source,
            artifact_path=wf_artifact,
            candidate_id=candidate_id,
            stage28_run_id=stage28_run_id,
            config_hash=config_hash,
            data_hash=data_hash,
            seed=seed,
        )
        + _evidence_records_for_gate(
            metric_payload=monte_carlo,
            metric_source_type=mc_source,
            artifact_path=mc_artifact,
            candidate_id=candidate_id,
            stage28_run_id=stage28_run_id,
            config_hash=config_hash,
            data_hash=data_hash,
            seed=seed,
        )
        + _evidence_records_for_gate(
            metric_payload=cross_seed,
            metric_source_type=cross_source,
            artifact_path=cross_artifact,
            candidate_id=candidate_id,
            stage28_run_id=stage28_run_id,
            config_hash=config_hash,
            data_hash=data_hash,
            seed=seed,
        )
    )

    evidence_quality = decision_evidence_guard(
        evidence_records,
        required_real_sources=required_real_sources,
        repo_root=Path("."),
    )
    payload = {
        "replay_metrics": replay,
        "walkforward_metrics": walkforward,
        "monte_carlo_metrics": monte_carlo,
        "cross_seed_metrics": cross_seed,
        "evidence_records": evidence_records,
        "evidence_quality": evidence_quality,
        "meta": {
            "source": "stage61_chain_writer_v2",
            "chain_id": str(chain_id),
            "stage28_run_id": str(stage28_run_id),
            "config_hash": str(config_hash),
            "data_hash": str(data_hash),
            "seed": int(seed),
            "source_types": {
                "replay": replay_source,
                "walkforward": wf_source,
                "monte_carlo": mc_source,
                "cross_seed": cross_source,
            },
        },
    }
    payload["summary_hash"] = stable_hash(payload, length=16)
    return {
        "status": "SUCCESS",
        "blocker_reason": "",
        "chain_metrics": payload,
    }
