"""Run Stage-58 transfer validation or scope exhaustion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.config import load_config
from buffmini.stage58 import assess_transfer_validation
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-58 transfer validation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(docs_dir: Path) -> str:
    stage57 = _load_json(docs_dir / "stage57_summary.json")
    suff = dict(stage57.get("metrics_sufficiency", {}))
    chain = _load_json(Path(str(suff.get("chain_metrics_path", ""))))
    run_id = str(dict(chain.get("meta", {})).get("stage28_run_id", "")).strip()
    if run_id:
        return run_id
    stage60 = _load_json(docs_dir / "stage60_summary.json")
    return str(stage60.get("stage28_run_id", "")).strip()


def main() -> None:
    args = parse_args()
    _cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage57_path = docs_dir / "stage57_summary.json"
    stage57 = json.loads(stage57_path.read_text(encoding="utf-8")) if stage57_path.exists() else {"verdict": "PARTIAL"}
    stage28_run_id = _resolve_stage28_run_id(docs_dir)
    transfer_artifact_path: Path | None = (Path(args.runs_dir) / stage28_run_id / "stage58" / "transfer_metrics_real.json") if stage28_run_id else None
    transfer_payload = (
        _load_json(transfer_artifact_path)
        if transfer_artifact_path is not None and transfer_artifact_path.exists() and transfer_artifact_path.is_file()
        else {}
    )
    transfer_metrics = dict(transfer_payload.get("metrics", {})) if transfer_payload else None
    transfer_source_type = str(transfer_payload.get("metric_source_type", "")).strip() if transfer_payload else ""
    result = assess_transfer_validation(
        stage57_verdict=str(stage57.get("verdict", "PARTIAL")),
        primary_metrics={"exp_lcb": float(stage57.get("replay_gate", {}).get("exp_lcb", 0.0))},
        transfer_metrics=transfer_metrics,
        transfer_metric_source_type=transfer_source_type,
        transfer_artifact_path=str(transfer_artifact_path or ""),
    )
    status = "SUCCESS" if str(result.get("verdict", "")) in {"WEAK_EDGE", "MEDIUM_EDGE"} else "PARTIAL"
    summary = {
        "stage": "58",
        "status": status,
        "execution_status": "EXECUTED" if stage57_path.exists() else "BLOCKED",
        "stage_role": "real_validation",
        "validation_state": "TRANSFER_CONFIRMED" if status == "SUCCESS" else "TRANSFER_NOT_CONFIRMED",
        "stage28_run_id": stage28_run_id,
        "stage57_verdict": str(stage57.get("verdict", "PARTIAL")),
        "transfer_result": result,
        "transfer_artifact_exists": bool(transfer_artifact_path is not None and transfer_artifact_path.exists() and transfer_artifact_path.is_file()),
        "summary_hash": stable_hash(
            {
                "status": status,
                "execution_status": "EXECUTED" if stage57_path.exists() else "BLOCKED",
                "validation_state": "TRANSFER_CONFIRMED" if status == "SUCCESS" else "TRANSFER_NOT_CONFIRMED",
                "stage28_run_id": stage28_run_id,
                "stage57_verdict": str(stage57.get("verdict", "PARTIAL")),
                "transfer_result": result,
                "transfer_artifact_exists": bool(transfer_artifact_path is not None and transfer_artifact_path.exists() and transfer_artifact_path.is_file()),
            },
            length=16,
        ),
    }
    summary_path = docs_dir / "stage58_summary.json"
    report_path = docs_dir / "stage58_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-58 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- stage57_verdict: `{summary['stage57_verdict']}`",
                f"- transfer_artifact_exists: `{summary['transfer_artifact_exists']}`",
                f"- transfer_result: `{summary['transfer_result']}`",
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
