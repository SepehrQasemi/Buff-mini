"""Run Stage-57 validation gates and verdict engine from Stage-52..56 chain artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage57 import PromotionGates, derive_stage57_verdict, detect_stale_inputs
from buffmini.validation import REAL_DECISION_SOURCE_TYPES, validate_metric_evidence_batch
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-57 verdict engine")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _empty_gates() -> dict[str, dict[str, Any]]:
    return {
        "replay_gate": {"passed": False, "trade_count": 0, "exp_lcb": 0.0, "maxDD": 1.0, "failure_reason_dominance": 1.0},
        "walkforward_gate": {"passed": False, "usable_windows": 0, "median_forward_exp_lcb": 0.0},
        "monte_carlo_gate": {"passed": False, "conservative_downside_bound": -1.0},
        "cross_seed_gate": {"passed": False, "surviving_seeds": 0},
    }


def _materialize_chain_metrics(docs_dir: Path, *, runs_dir: Path) -> dict[str, Any]:
    _ = runs_dir
    chain_metrics_path = docs_dir / "stage57_chain_metrics.json"
    return _load_json(chain_metrics_path)


def _validate_chain_metrics(payload: dict[str, Any]) -> tuple[bool, str]:
    required = (
        "replay_metrics",
        "walkforward_metrics",
        "monte_carlo_metrics",
        "cross_seed_metrics",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        return False, f"missing_chain_metrics_keys:{','.join(missing)}"
    source = str(dict(payload.get("meta", {})).get("source", "")).strip()
    allowed_sources = {
        "",
        "stage57_chain_metrics",
        "stage57_chain_metrics_v2",
        "stage61_chain_metrics",
        "stage61_chain_metrics_writer",
        "stage61_chain_writer_v2",
    }
    if source not in allowed_sources:
        return False, f"invalid_chain_metrics_source:{source}"
    evidence_records = payload.get("evidence_records", [])
    if not isinstance(evidence_records, list) or not evidence_records:
        return False, "missing_evidence_records"
    validated = validate_metric_evidence_batch([dict(item) for item in evidence_records if isinstance(item, dict)], repo_root=Path("."))
    if not bool(validated.get("valid", False)):
        return False, f"invalid_evidence_schema:{';'.join(validated.get('errors', []))}"
    return True, ""


def _derive_metrics_sufficiency(docs_dir: Path, *, runs_dir: Path) -> dict[str, Any]:
    stage52 = _load_json(docs_dir / "stage52_summary.json")
    stage53 = _load_json(docs_dir / "stage53_summary.json")
    stage54 = _load_json(docs_dir / "stage54_summary.json")
    stage55 = _load_json(docs_dir / "stage55_summary.json")
    stage56 = _load_json(docs_dir / "stage56_summary.json")
    chain_metrics_path = docs_dir / "stage57_chain_metrics.json"
    chain_metrics = _materialize_chain_metrics(docs_dir, runs_dir=runs_dir)

    missing_summaries = [
        name
        for name, payload in [
            ("stage52_summary", stage52),
            ("stage53_summary", stage53),
            ("stage54_summary", stage54),
            ("stage55_summary", stage55),
            ("stage56_summary", stage56),
        ]
        if not payload
    ]
    if missing_summaries:
        return {
            "valid": False,
            "reason": f"missing_chain_summaries:{','.join(missing_summaries)}",
            "chain_metrics_path": str(chain_metrics_path),
        }

    if not bool(stage53.get("quality_gate_passed", False)):
        return {
            "valid": False,
            "reason": "stage53_quality_gate_not_passed",
            "chain_metrics_path": str(chain_metrics_path),
        }
    if str(stage54.get("status", "PARTIAL")) != "SUCCESS":
        return {
            "valid": False,
            "reason": "stage54_not_success",
            "chain_metrics_path": str(chain_metrics_path),
        }
    if str(stage56.get("status", "PARTIAL")) != "SUCCESS":
        return {
            "valid": False,
            "reason": "stage56_not_success",
            "chain_metrics_path": str(chain_metrics_path),
        }

    is_valid, reason = _validate_chain_metrics(chain_metrics)
    if not is_valid:
        return {
            "valid": False,
            "reason": reason or "invalid_chain_metrics_contract",
            "chain_metrics_path": str(chain_metrics_path),
        }
    return {
        "valid": True,
        "reason": "",
        "chain_metrics_path": str(chain_metrics_path),
        "chain_metrics": chain_metrics,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(cfg.get("paths", {}).get("runs_dir", "runs"))

    source_paths = [
        docs_dir / "stage52_summary.json",
        docs_dir / "stage53_summary.json",
        docs_dir / "stage54_summary.json",
        docs_dir / "stage55_summary.json",
        docs_dir / "stage56_summary.json",
    ]
    chain_metrics_path = docs_dir / "stage57_chain_metrics.json"
    if chain_metrics_path.exists():
        source_paths.append(chain_metrics_path)
    stale_check = detect_stale_inputs(source_paths, max_age_hours=72.0)
    gates = PromotionGates(
        replay_trade_count_min=int(cfg.get("promotion_gates", {}).get("replay", {}).get("min_trade_count", 40)),
        replay_exp_lcb_min=float(cfg.get("promotion_gates", {}).get("replay", {}).get("min_exp_lcb", 0.0)),
        replay_maxdd_max=float(cfg.get("promotion_gates", {}).get("replay", {}).get("max_drawdown", 0.20)),
        replay_failure_dominance_max=float(cfg.get("promotion_gates", {}).get("replay", {}).get("max_failure_reason_dominance", 0.60)),
        walkforward_usable_windows_min=int(cfg.get("promotion_gates", {}).get("walkforward", {}).get("min_usable_windows", 5)),
        walkforward_median_exp_lcb_min=float(cfg.get("promotion_gates", {}).get("walkforward", {}).get("min_median_forward_exp_lcb", 0.0)),
        monte_carlo_downside_bound_min=float(cfg.get("promotion_gates", {}).get("monte_carlo", {}).get("min_downside_bound", 0.0)),
        cross_seed_survivors_min=int(cfg.get("promotion_gates", {}).get("cross_seed", {}).get("min_passing_seeds", 3)),
        required_real_sources=tuple(REAL_DECISION_SOURCE_TYPES) if bool(cfg.get("promotion_gates", {}).get("strict_real_evidence", True)) else tuple(),
    )

    sufficiency = _derive_metrics_sufficiency(docs_dir, runs_dir=runs_dir)
    history_path = docs_dir / "stage57_history.json"
    prior_history = _load_history(history_path)

    if stale_check["stale"]:
        gates_payload = _empty_gates()
        summary = {
            "stage": "57",
            "status": "PARTIAL",
            "execution_status": "BLOCKED_STALE_INPUTS",
            "stage_role": "real_validation",
            "validation_state": "STALE_INPUTS",
            "verdict": "STALE_INPUTS",
            "stale_inputs": stale_check,
            "metrics_sufficiency": sufficiency,
            "replay_gate": gates_payload["replay_gate"],
            "walkforward_gate": gates_payload["walkforward_gate"],
            "monte_carlo_gate": gates_payload["monte_carlo_gate"],
            "cross_seed_gate": gates_payload["cross_seed_gate"],
            "decision_evidence": {
                "allowed": False,
                "reason": "stale_inputs",
                "missing_real_sources": list(gates.required_real_sources),
            },
            "blocker_reason": "stale_inputs",
        }
    elif not bool(sufficiency.get("valid", False)):
        gates_payload = _empty_gates()
        summary = {
            "stage": "57",
            "status": "PARTIAL",
            "execution_status": "BLOCKED_INSUFFICIENT_INPUTS",
            "stage_role": "real_validation",
            "validation_state": "EVIDENCE_INSUFFICIENT",
            "verdict": "PARTIAL",
            "stale_inputs": stale_check,
            "metrics_sufficiency": sufficiency,
            "replay_gate": gates_payload["replay_gate"],
            "walkforward_gate": gates_payload["walkforward_gate"],
            "monte_carlo_gate": gates_payload["monte_carlo_gate"],
            "cross_seed_gate": gates_payload["cross_seed_gate"],
            "decision_evidence": {
                "allowed": False,
                "reason": "missing_or_invalid_chain_metrics",
                "missing_real_sources": list(gates.required_real_sources),
            },
            "blocker_reason": str(sufficiency.get("reason", "insufficient_chain_metrics")),
        }
    else:
        chain_metrics = dict(sufficiency.get("chain_metrics", {}))
        verdict = derive_stage57_verdict(
            replay_metrics=dict(chain_metrics.get("replay_metrics", {})),
            walkforward_metrics=dict(chain_metrics.get("walkforward_metrics", {})),
            monte_carlo_metrics=dict(chain_metrics.get("monte_carlo_metrics", {})),
            cross_seed_metrics=dict(chain_metrics.get("cross_seed_metrics", {})),
            evidence_records=[dict(item) for item in chain_metrics.get("evidence_records", []) if isinstance(item, dict)],
            validation_history=prior_history,
            gates=gates,
        )
        decision_evidence = dict(verdict.get("decision_evidence", {}))
        validation_state = "VALIDATED_EDGE" if verdict["verdict"] == "PASSING_EDGE" else ("NO_EDGE_IN_SCOPE" if verdict["verdict"] == "NO_EDGE_IN_SCOPE" else "NO_VALID_EDGE")
        if not bool(decision_evidence.get("allowed", True)):
            validation_state = "EVIDENCE_INSUFFICIENT"
        status = "SUCCESS" if bool(decision_evidence.get("allowed", True)) else "PARTIAL"
        summary = {
            "stage": "57",
            "status": status,
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": validation_state,
            "verdict": verdict["verdict"],
            "stale_inputs": stale_check,
            "metrics_sufficiency": sufficiency,
            "replay_gate": verdict["replay_gate"],
            "walkforward_gate": verdict["walkforward_gate"],
            "monte_carlo_gate": verdict["monte_carlo_gate"],
            "cross_seed_gate": verdict["cross_seed_gate"],
            "decision_evidence": decision_evidence,
            "blocker_reason": str(verdict.get("blocker_reason", "")),
        }
        history_entry = {"scope_frozen": True, "verdict": summary["verdict"]}
        history_path.write_text(json.dumps(prior_history + [history_entry], indent=2, allow_nan=False), encoding="utf-8")

    stale_hash_payload = {
        "stale": bool(summary["stale_inputs"].get("stale", False)),
        "missing_paths": list(summary["stale_inputs"].get("missing_paths", [])),
        "stale_paths": list(summary["stale_inputs"].get("stale_paths", [])),
        "max_age_hours": float(summary["stale_inputs"].get("max_age_hours", 72.0)),
    }
    summary["summary_hash"] = stable_hash(
        {
            "status": summary["status"],
            "execution_status": summary["execution_status"],
            "validation_state": summary["validation_state"],
            "verdict": summary["verdict"],
            "stale_inputs": stale_hash_payload,
            "metrics_sufficiency": summary["metrics_sufficiency"],
            "replay_gate": summary["replay_gate"],
            "walkforward_gate": summary["walkforward_gate"],
            "monte_carlo_gate": summary["monte_carlo_gate"],
            "cross_seed_gate": summary["cross_seed_gate"],
            "decision_evidence": summary["decision_evidence"],
            "blocker_reason": summary["blocker_reason"],
        },
        length=16,
    )
    summary_path = docs_dir / "stage57_summary.json"
    report_path = docs_dir / "stage57_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-57 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- verdict: `{summary['verdict']}`",
                f"- stale_inputs: `{summary['stale_inputs']}`",
                f"- metrics_sufficiency: `{summary['metrics_sufficiency']}`",
                f"- replay_gate: `{summary['replay_gate']}`",
                f"- walkforward_gate: `{summary['walkforward_gate']}`",
                f"- monte_carlo_gate: `{summary['monte_carlo_gate']}`",
                f"- cross_seed_gate: `{summary['cross_seed_gate']}`",
                f"- decision_evidence: `{summary['decision_evidence']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
