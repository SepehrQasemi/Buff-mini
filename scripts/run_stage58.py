"""Run Stage-58 transfer validation or scope exhaustion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.config import load_config
from buffmini.research.transfer import classify_transfer_outcome, discover_transfer_symbols
from buffmini.stage58 import assess_transfer_validation
from buffmini.utils.hashing import stable_hash
from buffmini.validation import compute_transfer_metrics, resolve_validation_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-58 transfer validation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(docs_dir: Path) -> str:
    stage57 = _load_json(docs_dir / "stage57_summary.json")
    suff = dict(stage57.get("metrics_sufficiency", {}))
    chain_path_raw = str(suff.get("chain_metrics_path", "")).strip()
    chain = _load_json(Path(chain_path_raw)) if chain_path_raw else {}
    run_id = str(dict(chain.get("meta", {})).get("stage28_run_id", "")).strip()
    if run_id:
        return run_id
    stage60 = _load_json(docs_dir / "stage60_summary.json")
    if str(stage60.get("stage28_run_id", "")).strip():
        return str(stage60.get("stage28_run_id", "")).strip()
    stage52 = _load_json(docs_dir / "stage52_summary.json")
    return str(stage52.get("stage28_run_id", "")).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage57_path = docs_dir / "stage57_summary.json"
    stage57 = json.loads(stage57_path.read_text(encoding="utf-8")) if stage57_path.exists() else {"verdict": "PARTIAL"}
    stage28_run_id = _resolve_stage28_run_id(docs_dir)
    transfer_artifact_path: Path | None = (Path(args.runs_dir) / stage28_run_id / "stage58" / "transfer_metrics_real.json") if stage28_run_id else None
    transfer_payload: dict[str, Any] = {}
    candidate = resolve_validation_candidate(
        runs_dir=Path(args.runs_dir),
        stage28_run_id=stage28_run_id,
        docs_dir=docs_dir,
    )
    primary_symbol = str(candidate.get("symbol", "")).strip()
    if not primary_symbol:
        universe = [str(v).strip() for v in cfg.get("universe", {}).get("symbols", []) if str(v).strip()]
        primary_symbol = universe[0] if universe else ""
    available_symbols = discover_transfer_symbols(cfg, primary_symbol=primary_symbol)
    preferred_transfer_symbol = str(cfg.get("research_scope", {}).get("expansion_rules", {}).get("transfer_symbol", "")).strip()
    alternative_symbols = [symbol for symbol in available_symbols if symbol != primary_symbol]
    transfer_symbol = preferred_transfer_symbol or (alternative_symbols[0] if alternative_symbols else "")
    transfer_matrix_path = (Path(args.runs_dir) / stage28_run_id / "stage58" / "transfer_matrix_real.json") if stage28_run_id else None
    transfer_matrix_rows: list[dict[str, Any]] = []
    if transfer_artifact_path is not None and candidate and transfer_symbol:
        transfer_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        primary_metrics = {
            "exp_lcb": float(stage57.get("replay_gate", {}).get("exp_lcb", 0.0)),
            "trade_count": int(stage57.get("replay_gate", {}).get("trade_count", 0)),
        }
        for symbol in alternative_symbols:
            transfer = compute_transfer_metrics(candidate=candidate, config=cfg, symbol=symbol)
            row = {
                "metric_source_type": "real_transfer",
                "artifact_path": str((Path(args.runs_dir) / stage28_run_id / "stage58" / f"transfer_{symbol.replace('/', '-')}_metrics_real.json")),
                "candidate_id": str(candidate.get("candidate_id", "")),
                "primary_symbol": primary_symbol,
                "transfer_symbol": symbol,
                "timeframe": str(candidate.get("timeframe", "")),
                "execution_status": str(transfer.get("execution_status", "BLOCKED")),
                "validation_state": str(transfer.get("validation_state", "TRANSFER_BLOCKED")),
                "decision_use_allowed": bool(transfer.get("decision_use_allowed", False)),
                "evidence_quality": "artifact_backed_real" if str(transfer.get("execution_status", "")) == "EXECUTED" else "real_but_blocked",
                "metrics": {
                    "trade_count": int(transfer.get("trade_count", 0)),
                    "exp_lcb": float(transfer.get("exp_lcb", 0.0)),
                    "maxDD": float(transfer.get("maxDD", 1.0)),
                },
                "market_meta": dict(transfer.get("market_meta", {})),
            }
            outcome = classify_transfer_outcome(primary_metrics=primary_metrics, transfer_metrics=row["metrics"])
            row["classification"] = str(outcome["classification"])
            row["diagnostics"] = list(outcome["diagnostics"])
            Path(row["artifact_path"]).write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
            transfer_matrix_rows.append(row)
        selected = next((row for row in transfer_matrix_rows if str(row["transfer_symbol"]) == transfer_symbol), transfer_matrix_rows[0] if transfer_matrix_rows else {})
        transfer_payload = dict(selected)
        if transfer_matrix_path is not None:
            transfer_matrix_path.write_text(json.dumps({"rows": transfer_matrix_rows}, indent=2, allow_nan=False), encoding="utf-8")
        transfer_artifact_path.write_text(json.dumps(transfer_payload, indent=2, allow_nan=False), encoding="utf-8")
    elif transfer_artifact_path is not None and transfer_artifact_path.exists() and transfer_artifact_path.is_file():
        transfer_payload = _load_json(transfer_artifact_path)
        if transfer_matrix_path is not None and transfer_matrix_path.exists():
            transfer_matrix_rows = list(_load_json(transfer_matrix_path).get("rows", []))
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
        "candidate_id": str(candidate.get("candidate_id", "")),
        "primary_symbol": primary_symbol,
        "transfer_symbol": transfer_symbol,
        "available_transfer_symbols": available_symbols,
        "stage57_verdict": str(stage57.get("verdict", "PARTIAL")),
        "transfer_result": result,
        "transfer_matrix": transfer_matrix_rows,
        "transfer_class_counts": {
            str(key): sum(1 for row in transfer_matrix_rows if str(row.get("classification", "")) == str(key))
            for key in sorted({str(row.get("classification", "")) for row in transfer_matrix_rows})
        },
        "transfer_execution_status": str(transfer_payload.get("execution_status", "NOT_ATTEMPTED")),
        "transfer_validation_state": str(transfer_payload.get("validation_state", "NOT_ATTEMPTED")),
        "decision_use_allowed": bool(transfer_payload.get("decision_use_allowed", False)),
        "evidence_quality": str(transfer_payload.get("evidence_quality", "missing")),
        "transfer_artifact_exists": bool(transfer_artifact_path is not None and transfer_artifact_path.exists() and transfer_artifact_path.is_file()),
        "transfer_matrix_artifact_exists": bool(transfer_matrix_path is not None and transfer_matrix_path.exists() and transfer_matrix_path.is_file()),
        "summary_hash": stable_hash(
            {
                "status": status,
                "execution_status": "EXECUTED" if stage57_path.exists() else "BLOCKED",
                "validation_state": "TRANSFER_CONFIRMED" if status == "SUCCESS" else "TRANSFER_NOT_CONFIRMED",
                "stage28_run_id": stage28_run_id,
                "candidate_id": str(candidate.get("candidate_id", "")),
                "primary_symbol": primary_symbol,
                "transfer_symbol": transfer_symbol,
                "available_transfer_symbols": available_symbols,
                "stage57_verdict": str(stage57.get("verdict", "PARTIAL")),
                "transfer_result": result,
                "transfer_matrix": transfer_matrix_rows,
                "transfer_class_counts": {
                    str(key): sum(1 for row in transfer_matrix_rows if str(row.get("classification", "")) == str(key))
                    for key in sorted({str(row.get("classification", "")) for row in transfer_matrix_rows})
                },
                "transfer_execution_status": str(transfer_payload.get("execution_status", "NOT_ATTEMPTED")),
                "transfer_validation_state": str(transfer_payload.get("validation_state", "NOT_ATTEMPTED")),
                "decision_use_allowed": bool(transfer_payload.get("decision_use_allowed", False)),
                "evidence_quality": str(transfer_payload.get("evidence_quality", "missing")),
                "transfer_artifact_exists": bool(transfer_artifact_path is not None and transfer_artifact_path.exists() and transfer_artifact_path.is_file()),
                "transfer_matrix_artifact_exists": bool(transfer_matrix_path is not None and transfer_matrix_path.exists() and transfer_matrix_path.is_file()),
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
                f"- candidate_id: `{summary['candidate_id']}`",
                f"- primary_symbol: `{summary['primary_symbol']}`",
                f"- transfer_symbol: `{summary['transfer_symbol']}`",
                f"- available_transfer_symbols: `{summary['available_transfer_symbols']}`",
                f"- stage57_verdict: `{summary['stage57_verdict']}`",
                f"- transfer_class_counts: `{summary['transfer_class_counts']}`",
                f"- transfer_execution_status: `{summary['transfer_execution_status']}`",
                f"- transfer_validation_state: `{summary['transfer_validation_state']}`",
                f"- decision_use_allowed: `{summary['decision_use_allowed']}`",
                f"- evidence_quality: `{summary['evidence_quality']}`",
                f"- transfer_artifact_exists: `{summary['transfer_artifact_exists']}`",
                f"- transfer_matrix_artifact_exists: `{summary['transfer_matrix_artifact_exists']}`",
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
