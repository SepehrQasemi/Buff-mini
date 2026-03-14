"""Run Stage-66 ML stack v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import math

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage66 import train_model_stack_v3
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-66 ML stack v3")
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


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage62_summary = _load_json(docs_dir / "stage62_summary.json")
    stage28_run_id = str(args.stage28_run_id).strip() or str(stage62_summary.get("stage28_run_id", "")).strip()
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-66")
    dataset_path = Path(args.runs_dir) / stage28_run_id / "stage62" / "training_dataset_v3.csv"
    if not dataset_path.exists():
        raise SystemExit(f"missing training_dataset_v3: {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    feature_columns = [str(v) for v in stage62_summary.get("feature_columns", []) if str(v).strip()]
    if not feature_columns:
        feature_columns = [col for col in dataset.columns if col not in {"timestamp", "candidate_id", "source_candidate_id", "tp_before_sl_label", "expected_net_after_cost_label", "mae_pct_label", "mfe_pct_label", "expected_hold_bars_label", "realized_label_present"}]
    usable_features: list[str] = []
    for col in feature_columns:
        if col not in dataset.columns:
            continue
        series = pd.to_numeric(dataset[col], errors="coerce")
        if int(series.notna().sum()) < 2:
            continue
        if int(series.nunique(dropna=True)) <= 1:
            continue
        dataset[col] = series.fillna(0.0)
        usable_features.append(col)
    feature_columns = usable_features
    try:
        if len(feature_columns) < 3:
            raise ValueError("insufficient_numeric_nonconstant_features_for_stage66")
        registry = train_model_stack_v3(
            dataset,
            feature_columns=feature_columns,
            seed=int(cfg.get("search", {}).get("seed", 42)),
        )
        status = "SUCCESS"
        blocker_reason = ""
    except Exception as exc:
        registry = {
            "version": "model_registry_v5",
            "seed": int(cfg.get("search", {}).get("seed", 42)),
            "feature_columns": feature_columns,
            "base_models": [],
            "optional_model_support": {"xgboost": False, "lightgbm": False, "catboost": False},
            "stacking": {"method": "mean_ensemble", "calibration": "platt"},
            "metrics": {"stack_logloss": 0.0, "stack_brier": 0.0, "train_summary_hash": ""},
            "summary_hash": "",
        }
        status = "PARTIAL"
        blocker_reason = str(exc)
    summary = {
        "stage": "66",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "ML_STACK_TRAINED_REPORTING_ONLY" if status == "SUCCESS" else "ML_STACK_PARTIAL_REPORTING_ONLY",
        "stage28_run_id": stage28_run_id,
        "base_models": registry["base_models"],
        "optional_model_support": registry["optional_model_support"],
        "used_feature_count": len(feature_columns),
        "stack_logloss": registry["metrics"]["stack_logloss"],
        "stack_brier": registry["metrics"]["stack_brier"],
        "model_registry_version": registry["version"],
        "decision_use_allowed": False,
        "blocker_reason": blocker_reason,
        "summary_hash": stable_hash(
            {
                "stage": "66",
                "status": status,
                "stage_role": "reporting_only",
                "validation_state": "ML_STACK_TRAINED_REPORTING_ONLY" if status == "SUCCESS" else "ML_STACK_PARTIAL_REPORTING_ONLY",
                "stage28_run_id": stage28_run_id,
                "base_models": registry["base_models"],
                "optional_model_support": registry["optional_model_support"],
                "used_feature_count": len(feature_columns),
                "stack_logloss": registry["metrics"]["stack_logloss"],
                "stack_brier": registry["metrics"]["stack_brier"],
                "model_registry_version": registry["version"],
                "decision_use_allowed": False,
                "registry_hash": registry.get("summary_hash", ""),
                "blocker_reason": blocker_reason,
            },
            length=16,
        ),
    }
    registry = _sanitize(registry)
    summary = _sanitize(summary)
    out_dir = Path(args.runs_dir) / stage28_run_id / "stage66"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_registry_v5.json").write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage66_model_registry_v5.json").write_text(json.dumps(registry, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage66_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage66_report.md").write_text(
        "\n".join(
            [
                "# Stage-66 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- base_models: `{summary['base_models']}`",
                f"- optional_model_support: `{summary['optional_model_support']}`",
                f"- used_feature_count: `{summary['used_feature_count']}`",
                f"- stack_logloss: `{summary['stack_logloss']}`",
                f"- stack_brier: `{summary['stack_brier']}`",
                f"- model_registry_version: `{summary['model_registry_version']}`",
                f"- decision_use_allowed: `{summary['decision_use_allowed']}`",
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
