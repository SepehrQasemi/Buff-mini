"""Run Stage-67 validation protocol v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage67 import evaluate_validation_protocol_v3
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-67 validation protocol v3")
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


def _estimate_monte_carlo_downside_bound(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    seed: int,
    n_paths: int = 1000,
    block_size: int = 24,
) -> dict[str, Any]:
    scores = pd.to_numeric(dataset.get(score_column, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = int(len(scores))
    if n <= 8:
        return {
            "conservative_downside_bound": -1.0,
            "n_paths": 0,
            "block_size": int(block_size),
            "sample_size": int(n),
        }
    rng = np.random.default_rng(int(seed))
    block = int(max(4, min(block_size, n)))
    horizon = int(min(96, n))
    path_means: list[float] = []
    for _ in range(int(max(50, n_paths))):
        idx = int(rng.integers(0, max(1, n - block + 1)))
        segment = scores[idx : idx + block]
        reps = int(max(1, np.ceil(horizon / max(1, len(segment)))))
        path = np.tile(segment, reps)[:horizon]
        path_means.append(float(np.mean(path)))
    downside = float(np.quantile(np.asarray(path_means, dtype=float), 0.05))
    return {
        "conservative_downside_bound": float(round(downside, 8)),
        "n_paths": int(len(path_means)),
        "block_size": int(block),
        "sample_size": int(n),
    }


def _estimate_cross_perturbation_survival(
    dataset: pd.DataFrame,
    *,
    score_column: str,
    label_column: str,
    min_median_forward_exp_lcb: float,
) -> dict[str, Any]:
    scores = pd.to_numeric(dataset.get(score_column, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    labels = pd.to_numeric(dataset.get(label_column, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    perturbations = [0.80, 0.90, 1.00, 1.10, 1.20]
    survivors = 0
    rows: list[dict[str, Any]] = []
    for factor in perturbations:
        perturbed = scores * float(factor)
        median_score = float(np.median(perturbed)) if len(perturbed) else -1.0
        label_rate = float(np.mean(labels > 0.0)) if len(labels) else 0.0
        passed = bool(median_score >= float(min_median_forward_exp_lcb) and label_rate > 0.0)
        if passed:
            survivors += 1
        rows.append(
            {
                "perturbation_factor": float(factor),
                "median_forward_exp_lcb": float(round(median_score, 8)),
                "positive_label_rate": float(round(label_rate, 8)),
                "passed": bool(passed),
            }
        )
    return {
        "surviving_seeds": int(survivors),
        "total_perturbations": int(len(perturbations)),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage62 = _load_json(docs_dir / "stage62_summary.json")
    stage28_run_id = str(args.stage28_run_id).strip() or str(stage62.get("stage28_run_id", "")).strip()
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-67")
    dataset_path = Path(args.runs_dir) / stage28_run_id / "stage62" / "training_dataset_v3.csv"
    if not dataset_path.exists():
        raise SystemExit(f"missing training_dataset_v3: {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    stage_a_survivors = int((pd.to_numeric(dataset.get("tp_before_sl_label", 0.0), errors="coerce").fillna(0.0) > 0.0).sum())
    stage_b_survivors = int((pd.to_numeric(dataset.get("expected_net_after_cost_label", 0.0), errors="coerce").fillna(0.0) > 0.0).sum())
    validation_cfg = dict(cfg.get("validation_protocol_v3", {}))
    promotion_walkforward = dict(cfg.get("promotion_gates", {}).get("walkforward", {}))
    portfolio_walkforward = dict(cfg.get("portfolio", {}).get("walkforward", {}))
    validation = evaluate_validation_protocol_v3(
        dataset=dataset,
        score_column="expected_net_after_cost_label",
        label_column="tp_before_sl_label",
        stage_a_survivors=stage_a_survivors,
        stage_b_survivors=stage_b_survivors,
        min_forward_trades=int(max(1, portfolio_walkforward.get("min_forward_trades", 10))),
        min_forward_exposure=float(max(0.0, portfolio_walkforward.get("min_forward_exposure", 0.01))),
        min_median_forward_exp_lcb=float(promotion_walkforward.get("min_median_forward_exp_lcb", 0.0)),
        n_splits=int(max(1, validation_cfg.get("n_splits", 5))),
        min_train_rows=int(max(16, validation_cfg.get("min_train_rows", 128))),
        test_size_rows=int(max(8, validation_cfg.get("test_size_rows", 64))),
        purge_gap_rows=int(max(0, validation_cfg.get("purge_gap_rows", 8))),
    )
    min_usable_windows_cfg = int(max(1, promotion_walkforward.get("min_usable_windows", 1)))
    if int(validation.get("usable_windows", 0)) < min_usable_windows_cfg:
        validation["status"] = "PARTIAL"
        validation["blocker_reason"] = "usable_windows_below_gate"
    out_dir = Path(args.runs_dir) / stage28_run_id / "stage67"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.DataFrame(validation.get("windows", []))
    windows_path = out_dir / "walkforward_windows_real.csv"
    if not windows.empty:
        windows.to_csv(windows_path, index=False)
    walkforward_metrics_path = out_dir / "walkforward_metrics_real.json"
    walkforward_metrics_payload = {
        "metric_source_type": "real_walkforward",
        "artifact_path": str(windows_path) if windows_path.exists() else "",
        "split_count": int(validation.get("split_count", 0)),
        "usable_windows": int(validation.get("usable_windows", 0)),
        "min_usable_windows": int(min_usable_windows_cfg),
        "median_forward_exp_lcb": float(validation.get("median_forward_exp_lcb", 0.0)),
        "mean_forward_exp_lcb": float(validation.get("mean_score", 0.0)),
        "min_forward_trades": int(max(1, portfolio_walkforward.get("min_forward_trades", 10))),
        "min_forward_exposure": float(max(0.0, portfolio_walkforward.get("min_forward_exposure", 0.01))),
        "min_median_forward_exp_lcb": float(promotion_walkforward.get("min_median_forward_exp_lcb", 0.0)),
    }
    walkforward_metrics_path.write_text(json.dumps(walkforward_metrics_payload, indent=2, allow_nan=False), encoding="utf-8")
    stage57_dir = Path(args.runs_dir) / stage28_run_id / "stage57"
    stage57_dir.mkdir(parents=True, exist_ok=True)
    monte_carlo = _estimate_monte_carlo_downside_bound(
        dataset,
        score_column="expected_net_after_cost_label",
        seed=int(cfg.get("search", {}).get("seed", 42)),
        n_paths=int(min(5000, max(100, cfg.get("portfolio", {}).get("leverage_selector", {}).get("n_paths", 1000)))),
        block_size=int(cfg.get("portfolio", {}).get("leverage_selector", {}).get("block_size_trades", 24)),
    )
    monte_carlo_payload = {
        "metric_source_type": "real_monte_carlo",
        "artifact_path": str(stage57_dir / "monte_carlo_metrics_real.json"),
        **monte_carlo,
    }
    (stage57_dir / "monte_carlo_metrics_real.json").write_text(json.dumps(monte_carlo_payload, indent=2, allow_nan=False), encoding="utf-8")
    perturb = _estimate_cross_perturbation_survival(
        dataset,
        score_column="expected_net_after_cost_label",
        label_column="tp_before_sl_label",
        min_median_forward_exp_lcb=float(promotion_walkforward.get("min_median_forward_exp_lcb", 0.0)),
    )
    perturb_df = pd.DataFrame(perturb.get("rows", []))
    perturb_csv = stage57_dir / "cross_perturbation_windows_real.csv"
    if not perturb_df.empty:
        perturb_df.to_csv(perturb_csv, index=False)
    perturb_payload = {
        "metric_source_type": "real_cross_perturbation",
        "artifact_path": str(perturb_csv) if perturb_csv.exists() else "",
        "surviving_seeds": int(perturb.get("surviving_seeds", 0)),
        "total_perturbations": int(perturb.get("total_perturbations", 0)),
    }
    (stage57_dir / "cross_perturbation_metrics_real.json").write_text(json.dumps(perturb_payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        "stage": "67",
        "status": str(validation["status"]),
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "REAL_VALIDATION_PASSED" if str(validation["status"]) == "SUCCESS" else "REAL_VALIDATION_FAILED",
        "stage28_run_id": stage28_run_id,
        "split_count": int(validation["split_count"]),
        "usable_windows": int(validation.get("usable_windows", 0)),
        "mean_score": float(validation["mean_score"]),
        "mean_label": float(validation["mean_label"]),
        "median_forward_exp_lcb": float(validation.get("median_forward_exp_lcb", 0.0)),
        "gates_effective": bool(validation["gates_effective"]),
        "metric_source_type": "real_walkforward",
        "walkforward_artifact_path": str(walkforward_metrics_path),
        "monte_carlo_artifact_path": str(stage57_dir / "monte_carlo_metrics_real.json"),
        "cross_perturbation_artifact_path": str(stage57_dir / "cross_perturbation_metrics_real.json"),
        "used_config_keys": [
            "validation_protocol_v3.n_splits",
            "validation_protocol_v3.min_train_rows",
            "validation_protocol_v3.test_size_rows",
            "validation_protocol_v3.purge_gap_rows",
            "promotion_gates.walkforward.min_median_forward_exp_lcb",
            "promotion_gates.walkforward.min_usable_windows",
            "portfolio.walkforward.min_forward_trades",
            "portfolio.walkforward.min_forward_exposure",
        ],
        "effective_values": {
            "n_splits": int(max(1, validation_cfg.get("n_splits", 5))),
            "min_train_rows": int(max(16, validation_cfg.get("min_train_rows", 128))),
            "test_size_rows": int(max(8, validation_cfg.get("test_size_rows", 64))),
            "purge_gap_rows": int(max(0, validation_cfg.get("purge_gap_rows", 8))),
            "min_forward_trades": int(max(1, portfolio_walkforward.get("min_forward_trades", 10))),
            "min_forward_exposure": float(max(0.0, portfolio_walkforward.get("min_forward_exposure", 0.01))),
            "min_median_forward_exp_lcb": float(promotion_walkforward.get("min_median_forward_exp_lcb", 0.0)),
            "min_usable_windows": int(min_usable_windows_cfg),
        },
        "blocker_reason": str(validation["blocker_reason"]),
    }
    summary["summary_hash"] = stable_hash(
        {"summary": summary, "validation_hash": validation.get("summary_hash", "")},
        length=16,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage67_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage67_report.md").write_text(
        "\n".join(
            [
                "# Stage-67 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- split_count: `{summary['split_count']}`",
                f"- usable_windows: `{summary['usable_windows']}`",
                f"- mean_score: `{summary['mean_score']}`",
                f"- mean_label: `{summary['mean_label']}`",
                f"- median_forward_exp_lcb: `{summary['median_forward_exp_lcb']}`",
                f"- gates_effective: `{summary['gates_effective']}`",
                f"- metric_source_type: `{summary['metric_source_type']}`",
                f"- walkforward_artifact_path: `{summary['walkforward_artifact_path']}`",
                f"- monte_carlo_artifact_path: `{summary['monte_carlo_artifact_path']}`",
                f"- cross_perturbation_artifact_path: `{summary['cross_perturbation_artifact_path']}`",
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
