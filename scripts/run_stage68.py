"""Run Stage-68 uncertainty-aware gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage68 import apply_uncertainty_gate_v3
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-68 uncertainty gating")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage62 = _load_json(docs_dir / "stage62_summary.json")
    stage67 = _load_json(docs_dir / "stage67_summary.json")
    stage28_run_id = str(args.stage28_run_id).strip() or str(stage62.get("stage28_run_id", "")).strip()
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-68")
    outcomes_path = Path(args.runs_dir) / stage28_run_id / "stage62" / "candidate_outcomes_v3.csv"
    if not outcomes_path.exists():
        raise SystemExit(f"missing candidate_outcomes_v3: {outcomes_path}")
    outcomes = pd.read_csv(outcomes_path)
    gate_cfg = dict(cfg.get("uncertainty_gate", {}))
    stage67_real_ok = (
        str(stage67.get("status", "")).upper() == "SUCCESS"
        and str(stage67.get("metric_source_type", "")).strip() == "real_walkforward"
    )
    gated = apply_uncertainty_gate_v3(
        outcomes,
        max_uncertainty=float(gate_cfg.get("max_uncertainty", 0.25)),
        min_tp_before_sl_prob=float(gate_cfg.get("min_tp_before_sl_prob", 0.55)),
        min_expected_net_after_cost=float(gate_cfg.get("min_expected_net_after_cost", 0.0)),
        allow_proxy_uncertainty=bool(gate_cfg.get("allow_proxy_uncertainty", False)),
    )
    gated_df = gated["gated"]
    counts = gated["counts"]
    if not stage67_real_ok:
        gated_df = gated_df.iloc[0:0].copy()
        counts = {"input": int(len(outcomes)), "accepted": 0, "abstained": int(len(outcomes))}
    status = "SUCCESS" if int(counts["accepted"]) > 0 and stage67_real_ok else "PARTIAL"
    out_dir = Path(args.runs_dir) / stage28_run_id / "stage68"
    out_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_metrics_path = out_dir / "uncertainty_gate_metrics.json"
    uncertainty_metrics_payload = {
        "metric_source_type": str(gated.get("uncertainty_source_type", "synthetic")),
        "allow_proxy_uncertainty": bool(gated.get("allow_proxy_uncertainty", False)),
        "input_candidates": int(counts["input"]),
        "accepted_candidates": int(counts["accepted"]),
        "abstained_candidates": int(counts["abstained"]),
        "stage67_real_validation_ok": bool(stage67_real_ok),
        "stage67_summary_hash": str(stage67.get("summary_hash", "")),
    }
    uncertainty_metrics_path.write_text(json.dumps(uncertainty_metrics_payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "stage": "68",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "heuristic_filter",
        "validation_state": "HEURISTIC_FILTER_PASSED" if status == "SUCCESS" else "HEURISTIC_FILTER_BLOCKED",
        "stage28_run_id": stage28_run_id,
        "input_candidates": int(counts["input"]),
        "accepted_candidates": int(counts["accepted"]),
        "abstained_candidates": int(counts["abstained"]),
        "metric_source_type": str(gated.get("uncertainty_source_type", "synthetic")),
        "uncertainty_artifact_path": str(uncertainty_metrics_path),
        "stage67_real_validation_ok": bool(stage67_real_ok),
        "used_config_keys": [
            "uncertainty_gate.max_uncertainty",
            "uncertainty_gate.min_tp_before_sl_prob",
            "uncertainty_gate.min_expected_net_after_cost",
            "uncertainty_gate.allow_proxy_uncertainty",
        ],
        "effective_values": {
            "max_uncertainty": float(gate_cfg.get("max_uncertainty", 0.25)),
            "min_tp_before_sl_prob": float(gate_cfg.get("min_tp_before_sl_prob", 0.55)),
            "min_expected_net_after_cost": float(gate_cfg.get("min_expected_net_after_cost", 0.0)),
            "allow_proxy_uncertainty": bool(gate_cfg.get("allow_proxy_uncertainty", False)),
        },
        "blocker_reason": "" if status == "SUCCESS" else ("stage67_not_real_or_not_success" if not stage67_real_ok else "all_candidates_abstained"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    gated_df.to_csv(out_dir / "gated_candidates.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage68_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage68_report.md").write_text(
        "\n".join(
            [
                "# Stage-68 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- input_candidates: `{summary['input_candidates']}`",
                f"- accepted_candidates: `{summary['accepted_candidates']}`",
                f"- abstained_candidates: `{summary['abstained_candidates']}`",
                f"- metric_source_type: `{summary['metric_source_type']}`",
                f"- uncertainty_artifact_path: `{summary['uncertainty_artifact_path']}`",
                f"- stage67_real_validation_ok: `{summary['stage67_real_validation_ok']}`",
                f"- used_config_keys: `{summary['used_config_keys']}`",
                f"- effective_values: `{summary['effective_values']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
