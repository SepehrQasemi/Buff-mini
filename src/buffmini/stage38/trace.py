"""Deterministic runtime trace helpers for Stage-38."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class TraceEvent:
    """One deterministic execution-trace event."""

    order: int
    stage: str
    component: str
    action: str
    input_rows: int
    output_rows: int
    details: dict[str, Any]
    artifact_paths: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": int(self.order),
            "stage": str(self.stage),
            "component": str(self.component),
            "action": str(self.action),
            "input_rows": int(self.input_rows),
            "output_rows": int(self.output_rows),
            "details": _json_safe(self.details),
            "artifact_paths": [str(v) for v in self.artifact_paths],
        }


def build_stage28_execution_trace(*, stage28_dir: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Build deterministic, evidence-backed trace for one Stage-28 run."""

    run_dir = Path(stage28_dir)
    summary = _load_json(run_dir / "summary.json")
    stage_a = _safe_read_csv(run_dir / "selected_candidates_stageA.csv")
    stage_b = _safe_read_csv(run_dir / "selected_candidates_stageB.csv")
    stage_c = _safe_read_csv(run_dir / "finalists_stageC.csv")
    matrix_df = _safe_read_csv(run_dir / "context_matrix.csv")
    policy_trace = _safe_read_csv(run_dir / "policy_trace.csv")
    shadow_rejects = _safe_read_csv(run_dir / "shadow_live_rejects.csv")
    envelope = _safe_read_csv(run_dir / "feasibility_envelope.csv")
    usability = _safe_read_csv(run_dir / "usability_trace.csv")
    policy = _load_json(run_dir / "policy.json")

    active_series = _safe_active_candidates(policy_trace)
    net_score = pd.to_numeric(policy_trace.get("net_score", 0.0), errors="coerce").fillna(0.0)
    final_signal = pd.to_numeric(policy_trace.get("final_signal", 0), errors="coerce").fillna(0).astype(int)

    candidate_rows = int(active_series.str.len().gt(0).sum())
    nonzero_net = int(net_score.ne(0.0).sum())
    final_nonzero = int(final_signal.ne(0).sum())

    policy_contexts = dict(policy.get("contexts", {}))
    policy_context_count = int(len(policy_contexts))
    policy_candidate_count = int(
        sum(len(list((policy_contexts.get(ctx, {}) or {}).get("candidates", []))) for ctx in policy_contexts)
    )

    events = [
        TraceEvent(
            order=1,
            stage="entrypoint",
            component="scripts/run_stage28.py",
            action="cli_invoked",
            input_rows=0,
            output_rows=0,
            details={
                "config_path": str(config_path) if config_path else "",
                "seed": int(summary.get("seed", 0)),
                "mode": str(summary.get("mode", "")),
                "timeframes": list(summary.get("timeframes", [])),
                "symbols": list(summary.get("used_symbols", [])),
            },
            artifact_paths=[],
        ),
        TraceEvent(
            order=2,
            stage="config",
            component="buffmini.config.load_config",
            action="config_and_snapshot_resolved",
            input_rows=0,
            output_rows=0,
            details={
                "config_hash": str(summary.get("config_hash", "")),
                "data_snapshot_id": str(summary.get("data_snapshot_id", "")),
                "data_snapshot_hash": str(summary.get("data_snapshot_hash", "")),
                "coverage_gate_status": str(summary.get("coverage_gate_status", "")),
            },
            artifact_paths=[str((run_dir / "summary.json").as_posix())],
        ),
        TraceEvent(
            order=3,
            stage="feature_and_context",
            component="stage28.context_discovery + stage28.budget_funnel",
            action="candidate_matrix_and_funnel_selection",
            input_rows=int(matrix_df.shape[0]),
            output_rows=int(stage_c.shape[0]),
            details={
                "matrix_rows": int(matrix_df.shape[0]),
                "stage_a_rows": int(stage_a.shape[0]),
                "stage_b_rows": int(stage_b.shape[0]),
                "stage_c_rows": int(stage_c.shape[0]),
                "qualified_finalists": int(summary.get("qualified_finalists", 0)),
            },
            artifact_paths=[
                str((run_dir / "context_matrix.csv").as_posix()),
                str((run_dir / "selected_candidates_stageA.csv").as_posix()),
                str((run_dir / "selected_candidates_stageB.csv").as_posix()),
                str((run_dir / "finalists_stageC.csv").as_posix()),
            ],
        ),
        TraceEvent(
            order=4,
            stage="policy",
            component="stage28.policy_v2.build_policy_v2",
            action="policy_composed",
            input_rows=int(stage_c.shape[0]),
            output_rows=int(policy_context_count),
            details={
                "policy_context_count": policy_context_count,
                "policy_candidate_count": policy_candidate_count,
                "conflict_mode": str(policy.get("conflict_mode", "")),
                "warnings": list(policy.get("warnings", [])),
            },
            artifact_paths=[str((run_dir / "policy.json").as_posix()), str((run_dir / "policy_spec.md").as_posix())],
        ),
        TraceEvent(
            order=5,
            stage="composer",
            component="stage28.policy_v2.compose_policy_signal_v2",
            action="candidate_signals_composed_to_final_signal",
            input_rows=int(policy_trace.shape[0]),
            output_rows=int(final_nonzero),
            details={
                "policy_trace_rows": int(policy_trace.shape[0]),
                "candidate_rows_active": int(candidate_rows),
                "net_score_nonzero_rows": int(nonzero_net),
                "final_signal_nonzero_rows": int(final_nonzero),
            },
            artifact_paths=[str((run_dir / "policy_trace.csv").as_posix())],
        ),
        TraceEvent(
            order=6,
            stage="constraints",
            component="run_stage28._apply_live_constraints",
            action="live_constraints_and_rejects",
            input_rows=int(final_nonzero),
            output_rows=max(0, int(final_nonzero - shadow_rejects.shape[0])),
            details={
                "shadow_reject_rows": int(shadow_rejects.shape[0]),
                "shadow_reject_rate": float(summary.get("shadow_live_reject_rate", 0.0)),
                "top_reject_reasons": dict(summary.get("shadow_live_top_reasons", {})),
            },
            artifact_paths=[str((run_dir / "shadow_live_rejects.csv").as_posix())],
        ),
        TraceEvent(
            order=7,
            stage="evaluation",
            component="run_backtest + stage28 metrics",
            action="research_live_metrics_scored",
            input_rows=int(final_nonzero),
            output_rows=int(
                float(((summary.get("policy_metrics", {}) or {}).get("live", {}) or {}).get("trade_count", 0.0))
            ),
            details={
                "wf_executed_pct": float(summary.get("wf_executed_pct", 0.0)),
                "mc_trigger_pct": float(summary.get("mc_trigger_pct", 0.0)),
                "research": dict(((summary.get("policy_metrics", {}) or {}).get("research", {}) or {})),
                "live": dict(((summary.get("policy_metrics", {}) or {}).get("live", {}) or {})),
                "next_bottleneck": str(summary.get("next_bottleneck", "")),
                "verdict": str(summary.get("verdict", "")),
            },
            artifact_paths=[
                str((run_dir / "summary.json").as_posix()),
                str((run_dir / "feasibility_envelope.csv").as_posix()),
                str((run_dir / "usability_trace.csv").as_posix()),
            ],
        ),
    ]

    payload = {
        "stage": "38.1",
        "run_id": str(summary.get("run_id", "")),
        "stage28_dir": str(run_dir.as_posix()),
        "trace_events": [event.as_dict() for event in events],
        "artifact_row_counts": {
            "context_matrix": int(matrix_df.shape[0]),
            "selected_candidates_stageA": int(stage_a.shape[0]),
            "selected_candidates_stageB": int(stage_b.shape[0]),
            "finalists_stageC": int(stage_c.shape[0]),
            "policy_trace": int(policy_trace.shape[0]),
            "shadow_live_rejects": int(shadow_rejects.shape[0]),
            "feasibility_envelope": int(envelope.shape[0]),
            "usability_trace": int(usability.shape[0]),
        },
        "entrypoints": [
            {
                "name": "Streamlit Run Button",
                "flow": [
                    "launch_app.py",
                    "src/buffmini/ui/pages/20_strategy_lab.py",
                    "src/buffmini/ui/components/run_exec.py::start_pipeline",
                    "scripts/run_pipeline.py",
                ],
            },
            {
                "name": "Direct CLI",
                "flow": ["scripts/run_stage28.py", "scripts/stage37_activation_hunt.py", "scripts/run_stage37.py"],
            },
        ],
    }
    payload["trace_hash"] = trace_payload_hash(payload)
    return payload


def trace_payload_hash(payload: dict[str, Any]) -> str:
    """Hash trace payload deterministically, excluding volatile fields."""

    base = dict(payload)
    base.pop("trace_hash", None)
    return stable_hash(base, length=16)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


def _load_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_active_candidates(policy_trace: pd.DataFrame) -> pd.Series:
    if "active_candidates" not in policy_trace.columns or policy_trace.empty:
        return pd.Series("", index=policy_trace.index if not policy_trace.empty else pd.RangeIndex(0), dtype=str)
    raw = policy_trace["active_candidates"].copy()
    raw = raw.where(raw.notna(), "")
    out = raw.astype(str).replace({"nan": "", "None": ""})
    return out.fillna("")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
