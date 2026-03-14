"""Run Stage-61 chain metrics writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage60 import assess_chain_integrity
from buffmini.stage61 import materialize_stage57_chain_metrics
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-61 chain metrics writer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    budget_mode = str(cfg.get("budget_mode", {}).get("selected", "search"))

    stage60 = assess_chain_integrity(docs_dir=docs_dir, runs_dir=runs_dir, budget_mode_selected=budget_mode)
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    stage52 = _load_json(docs_dir / "stage52_summary.json")
    data_hash = str(stage39.get("data_hash", stage52.get("data_hash", ""))).strip() or "unknown_data_hash"
    config_hash = compute_config_hash(cfg)
    strict_real_evidence = bool(cfg.get("promotion_gates", {}).get("strict_real_evidence", True))
    required_real_sources = (
        [
            "real_replay",
            "real_walkforward",
            "real_monte_carlo",
            "real_cross_perturbation",
        ]
        if strict_real_evidence
        else []
    )
    result = materialize_stage57_chain_metrics(
        docs_dir=docs_dir,
        runs_dir=runs_dir,
        stage28_run_id=str(stage60.get("stage28_run_id", "")),
        chain_id=str(stage60.get("chain_id", "")),
        config_hash=config_hash,
        data_hash=data_hash,
        seed=int(cfg.get("search", {}).get("seed", 42)),
        required_real_sources=required_real_sources,
    )
    if result["status"] == "SUCCESS":
        chain_path = docs_dir / "stage57_chain_metrics.json"
        chain_path.write_text(json.dumps(result["chain_metrics"], indent=2, allow_nan=False), encoding="utf-8")
    evidence_quality = dict(result.get("chain_metrics", {}).get("evidence_quality", {}))
    summary_status = "SUCCESS" if bool(result["status"] == "SUCCESS") and bool(evidence_quality.get("allowed", False)) else "PARTIAL"
    summary = {
        "stage": "61",
        "status": summary_status,
        "execution_status": "EXECUTED" if str(result["status"]) == "SUCCESS" else "BLOCKED",
        "stage_role": "orchestration_only",
        "stage28_run_id": str(stage60.get("stage28_run_id", "")),
        "chain_id": str(stage60.get("chain_id", "")),
        "wrote_chain_metrics": bool(result["status"] == "SUCCESS"),
        "chain_metrics_source": str(result.get("chain_metrics", {}).get("meta", {}).get("source", "")),
        "decision_evidence_allowed": bool(evidence_quality.get("allowed", False)),
        "strict_real_evidence": bool(strict_real_evidence),
        "missing_real_sources": list(evidence_quality.get("missing_real_sources", [])),
        "blocked_decision_metrics": list(evidence_quality.get("blocked_decision_metrics", [])),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "blocker_reason": str(result.get("blocker_reason", "")) or ("" if bool(evidence_quality.get("allowed", False)) else "decision_evidence_not_ready"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)

    summary_path = docs_dir / "stage61_summary.json"
    report_path = docs_dir / "stage61_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-61 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- chain_id: `{summary['chain_id']}`",
                f"- wrote_chain_metrics: `{summary['wrote_chain_metrics']}`",
                f"- chain_metrics_source: `{summary['chain_metrics_source']}`",
                f"- decision_evidence_allowed: `{summary['decision_evidence_allowed']}`",
                f"- strict_real_evidence: `{summary['strict_real_evidence']}`",
                f"- missing_real_sources: `{summary['missing_real_sources']}`",
                f"- blocked_decision_metrics: `{summary['blocked_decision_metrics']}`",
                f"- config_hash: `{summary['config_hash']}`",
                f"- data_hash: `{summary['data_hash']}`",
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
