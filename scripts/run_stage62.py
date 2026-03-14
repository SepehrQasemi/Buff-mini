"""Run Stage-62 candidate-realized labels v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage62 import build_candidate_outcomes_v3, build_training_dataset_v3, evaluate_quality_gate_v3
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-62 candidate-realized labels")
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


def _resolve_stage28_run_id(*, args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage60 = _load_json(docs_dir / "stage60_summary.json")
    if str(stage60.get("stage28_run_id", "")).strip():
        return str(stage60.get("stage28_run_id", "")).strip()
    stage52 = _load_json(docs_dir / "stage52_summary.json")
    return str(stage52.get("stage28_run_id", "")).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    stage28_run_id = _resolve_stage28_run_id(args=args, docs_dir=docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-62")

    base = runs_dir / stage28_run_id
    stage52 = _read_csv(base / "stage52" / "setup_candidates_v2.csv")
    stage48_a = _read_csv(base / "stage48" / "stage48_stage_a_survivors.csv")
    stage48_b = _read_csv(base / "stage48" / "stage48_stage_b_survivors.csv")
    stage48_ranked = _read_csv(base / "stage48" / "stage48_ranked_candidates.csv")
    stage53_pred = _read_csv(base / "stage53" / "predictions.csv")
    stage53_summary = _load_json(docs_dir / "stage53_summary.json")

    outcomes = build_candidate_outcomes_v3(
        stage52_candidates=stage52,
        stage48_stage_a=stage48_a,
        stage48_stage_b=stage48_b,
        stage53_predictions=stage53_pred,
        stage48_ranked=stage48_ranked,
    )
    feature_cols = [str(v) for v in stage53_summary.get("feature_columns", []) if str(v).strip()]
    numeric_cols = [col for col in outcomes.columns if pd.api.types.is_numeric_dtype(outcomes[col])]
    numeric_candidates = [col for col in numeric_cols if not str(col).endswith("_label") and col not in {"realized_label_present"}]
    if feature_cols:
        for col in numeric_candidates:
            if col not in feature_cols:
                feature_cols.append(col)
    else:
        feature_cols = numeric_candidates
    dataset = build_training_dataset_v3(outcomes, feature_columns=feature_cols)
    quality = evaluate_quality_gate_v3(dataset, feature_columns=feature_cols)

    status = "SUCCESS" if bool(quality["passed"]) else "PARTIAL"
    summary = {
        "stage": "62",
        "status": status,
        "stage28_run_id": stage28_run_id,
        "candidate_outcomes_rows": int(len(outcomes)),
        "training_dataset_rows": int(len(dataset)),
        "feature_columns": feature_cols,
        "quality_gate_passed": bool(quality["passed"]),
        "label_coverage": float(quality["label_coverage"]),
        "non_constant_feature_count": int(quality["non_constant_feature_count"]),
        "quality_gate_reason": str(quality["reason"]),
        "blocker_reason": str("" if quality["passed"] else quality["reason"]),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)

    out_dir = base / "stage62"
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(out_dir / "candidate_outcomes_v3.csv", index=False)
    dataset.to_csv(out_dir / "training_dataset_v3.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage62_summary.json"
    report_path = docs_dir / "stage62_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-62 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- candidate_outcomes_rows: `{summary['candidate_outcomes_rows']}`",
                f"- training_dataset_rows: `{summary['training_dataset_rows']}`",
                f"- quality_gate_passed: `{summary['quality_gate_passed']}`",
                f"- label_coverage: `{summary['label_coverage']}`",
                f"- non_constant_feature_count: `{summary['non_constant_feature_count']}`",
                f"- quality_gate_reason: `{summary['quality_gate_reason']}`",
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
