"""Full run trace reporting for end-to-end audits."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.utils.hashing import stable_hash


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload


def _collect_docs_summaries(docs_dir: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for path in docs_dir.glob("*_summary.json"):
        payload = _load_json(path)
        summaries[path.name] = payload
    extra = docs_dir / "stage57_chain_metrics.json"
    if extra.exists():
        summaries[extra.name] = _load_json(extra)
    return summaries


def _collect_run_summaries(runs_dir: Path, stage28_run_id: str) -> dict[str, Any]:
    if not stage28_run_id:
        return {}
    base = runs_dir / stage28_run_id
    if not base.exists():
        return {}
    run_summaries: dict[str, Any] = {}
    for stage_dir in base.iterdir():
        if not stage_dir.is_dir():
            continue
        summary_path = stage_dir / "summary.json"
        if summary_path.exists():
            run_summaries[stage_dir.name] = _load_json(summary_path)
    return run_summaries


def _resolve_stage28_run_id(docs_summaries: dict[str, Any]) -> str:
    for key in (
        "stage60_summary.json",
        "stage52_summary.json",
        "stage39_signal_generation_summary.json",
        "stage28_master_summary.json",
    ):
        payload = docs_summaries.get(key)
        if isinstance(payload, dict):
            if key == "stage28_master_summary.json":
                value = str(payload.get("run_id", "")).strip()
            else:
                value = str(payload.get("stage28_run_id", "")).strip()
            if value:
                return value
    return ""


def _resolve_chain_id(docs_summaries: dict[str, Any]) -> str:
    payload = docs_summaries.get("stage60_summary.json")
    if isinstance(payload, dict):
        value = str(payload.get("chain_id", "")).strip()
        if value:
            return value
    return ""


def _extract_stage_sequence(docs_summaries: dict[str, Any]) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for name, payload in sorted(docs_summaries.items()):
        if not name.startswith("stage") or not name.endswith("_summary.json"):
            continue
        if not isinstance(payload, dict):
            continue
        stage = str(payload.get("stage", "")).strip()
        status = str(payload.get("status", "")).strip()
        summary_hash = str(payload.get("summary_hash", "")).strip()
        if stage:
            sequence.append(
                {"stage": stage, "status": status, "summary_hash": summary_hash}
            )
    return sequence


def _derive_zero_reasons(stage57_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage57_summary, dict):
        return ["stage57_summary_missing"]
    status = str(stage57_summary.get("status", "PARTIAL"))
    if status != "SUCCESS":
        return [f"stage57_status:{status}"]
    reasons: list[str] = []
    replay = dict(stage57_summary.get("replay_gate", {}))
    walk = dict(stage57_summary.get("walkforward_gate", {}))
    mc = dict(stage57_summary.get("monte_carlo_gate", {}))
    cs = dict(stage57_summary.get("cross_seed_gate", {}))
    if not bool(replay.get("passed", False)):
        reasons.append(
            "replay_gate_failed:"
            f"trade_count={replay.get('trade_count')},"
            f"exp_lcb={replay.get('exp_lcb')},"
            f"maxDD={replay.get('maxDD')},"
            f"failure_dom={replay.get('failure_reason_dominance')}"
        )
    if not bool(walk.get("passed", False)):
        reasons.append(
            "walkforward_gate_failed:"
            f"usable_windows={walk.get('usable_windows')},"
            f"median_forward_exp_lcb={walk.get('median_forward_exp_lcb')}"
        )
    if not bool(mc.get("passed", False)):
        reasons.append(
            "monte_carlo_gate_failed:"
            f"conservative_downside_bound={mc.get('conservative_downside_bound')}"
        )
    if not bool(cs.get("passed", False)):
        reasons.append(
            "cross_seed_gate_failed:"
            f"surviving_seeds={cs.get('surviving_seeds')}"
        )
    return reasons


def _extract_evidence_quality(docs_summaries: dict[str, Any]) -> dict[str, Any]:
    stage57 = docs_summaries.get("stage57_summary.json")
    chain = docs_summaries.get("stage57_chain_metrics.json")
    if not isinstance(stage57, dict):
        return {"decision_evidence_allowed": False, "missing_real_sources": ["stage57_summary_missing"]}
    evidence = dict(stage57.get("decision_evidence", {}))
    source_types = {}
    if isinstance(chain, dict):
        source_types = dict((chain.get("meta", {}) or {}).get("source_types", {}))
    return {
        "decision_evidence_allowed": bool(evidence.get("allowed", False)),
        "missing_real_sources": list(evidence.get("missing_real_sources", [])),
        "blocked_decision_metrics": list(evidence.get("blocked_decision_metrics", [])),
        "source_types": source_types,
    }


def build_full_trace_report(
    *,
    docs_dir: Path,
    runs_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    runs_dir = Path(runs_dir)
    cfg = load_config(Path(config_path))
    docs_summaries = _collect_docs_summaries(docs_dir)
    stage28_run_id = _resolve_stage28_run_id(docs_summaries)
    chain_id = _resolve_chain_id(docs_summaries)
    run_summaries = _collect_run_summaries(runs_dir, stage28_run_id)
    stage_sequence = _extract_stage_sequence(docs_summaries)
    stage57_summary = docs_summaries.get("stage57_summary.json")
    zero_reasons = _derive_zero_reasons(stage57_summary if isinstance(stage57_summary, dict) else None)
    evidence_quality = _extract_evidence_quality(docs_summaries)
    reproducibility = dict(cfg.get("reproducibility", {}))

    parameters = {
        "research_scope": cfg.get("research_scope", {}),
        "budget_mode": cfg.get("budget_mode", {}),
        "promotion_gates": cfg.get("promotion_gates", {}),
        "tradability_model": cfg.get("tradability_model", {}),
        "model_stack_v3": cfg.get("model_stack_v3", {}),
        "validation_protocol_v3": cfg.get("validation_protocol_v3", {}),
        "uncertainty_gate": cfg.get("uncertainty_gate", {}),
        "campaign_memory": cfg.get("campaign_memory", {}),
        "data_sources": cfg.get("data_sources", {}),
        "reproducibility": reproducibility,
    }

    payload = {
        "trace_version": "v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_hash": stable_hash(cfg, length=16),
        "stage28_run_id": stage28_run_id,
        "chain_id": chain_id,
        "parameters": parameters,
        "docs_summaries": docs_summaries,
        "run_summaries": run_summaries,
        "stage_sequence": stage_sequence,
        "zero_reasons": zero_reasons,
        "evidence_quality": evidence_quality,
    }
    payload["summary_hash"] = stable_hash(
        {
            "config_hash": payload["config_hash"],
            "stage28_run_id": stage28_run_id,
            "chain_id": chain_id,
            "stage_sequence": stage_sequence,
            "zero_reasons": zero_reasons,
            "evidence_quality": evidence_quality,
        },
        length=16,
    )
    return payload


def write_full_trace_report(
    *,
    docs_dir: Path,
    runs_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    payload = build_full_trace_report(docs_dir=docs_dir, runs_dir=runs_dir, config_path=config_path)
    json_path = docs_dir / "full_trace_summary.json"
    md_path = docs_dir / "full_trace_report.md"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = [
        "# Full Trace Report",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- stage28_run_id: `{payload['stage28_run_id']}`",
        f"- chain_id: `{payload['chain_id']}`",
        f"- config_hash: `{payload['config_hash']}`",
        f"- summary_hash: `{payload['summary_hash']}`",
        "",
        "## Zero Reasons",
    ]
    if payload["zero_reasons"]:
        for reason in payload["zero_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Evidence Quality")
    lines.append(f"- decision_evidence_allowed: `{payload.get('evidence_quality', {}).get('decision_evidence_allowed')}`")
    lines.append(f"- missing_real_sources: `{payload.get('evidence_quality', {}).get('missing_real_sources', [])}`")
    lines.append(f"- blocked_decision_metrics: `{payload.get('evidence_quality', {}).get('blocked_decision_metrics', [])}`")
    lines.append(f"- source_types: `{payload.get('evidence_quality', {}).get('source_types', {})}`")
    lines.append("")
    lines.append("## Stage Sequence")
    for item in payload["stage_sequence"]:
        lines.append(
            f"- stage {item.get('stage')}: status={item.get('status')} hash={item.get('summary_hash')}"
        )
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return payload
