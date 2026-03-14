"""Run Stage-71 replay engine acceleration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage71 import measure_replay_acceleration
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-71 replay acceleration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_returns(docs_dir: Path) -> np.ndarray:
    path = docs_dir / "stage65_features_v3.csv"
    if path.exists():
        frame = pd.read_csv(path)
        if "ret_1" in frame.columns:
            return pd.to_numeric(frame["ret_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    grid = np.arange(5000, dtype=float)
    return np.sin(grid / 23.0) * 0.001 + np.cos(grid / 47.0) * 0.0007


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage70 = _load_json(docs_dir / "stage70_summary.json")
    candidates_path = docs_dir / "stage70_expanded_candidates.csv"
    if not candidates_path.exists():
        raise SystemExit(f"missing expanded candidates: {candidates_path}")
    candidates = pd.read_csv(candidates_path).head(3000).copy()
    returns = _safe_returns(docs_dir)

    metrics = measure_replay_acceleration(
        candidates=candidates,
        returns=returns,
        data_hash=stable_hash({"len_returns": int(len(returns))}, length=12),
        setup_signature=stable_hash({"candidate_count": int(len(candidates))}, length=12),
        timeframe="1h",
        cost_model="simple_v1",
        scope_id=str(stage70.get("summary_hash", "")),
    )
    consistency_delta = float(metrics.get("consistency_delta", 0.0))
    consistency_ok = bool(consistency_delta <= 1e-6)
    frozen_mode = bool(cfg.get("reproducibility", {}).get("frozen_research_mode", False))
    summary = {
        "stage": "71",
        "status": "SUCCESS" if str(metrics.get("status", "PARTIAL")) == "SUCCESS" and consistency_ok else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "MEASURED_RUNTIME" if consistency_ok else "MEASURED_RUNTIME_INCONSISTENT",
        "candidate_count_measured": int(len(candidates)),
        "baseline_runtime_seconds": float(metrics.get("baseline_runtime_seconds", 0.0)),
        "optimized_runtime_seconds": float(metrics.get("optimized_runtime_seconds", 0.0)),
        "speedup_pct": float(metrics.get("speedup_pct", 0.0)),
        "meets_target_40pct": bool(metrics.get("meets_target_40pct", False)),
        "measurement_type": "measured_runtime_probe",
        "projected_only": False,
        "frozen_research_mode": bool(frozen_mode),
        "determinism_assumptions": {
            "candidate_seed_hash": "stable_hash",
            "python_hash_randomization_independent": True,
        },
        "cache_key": str(metrics.get("cache_key", "")),
        "consistency_delta": float(consistency_delta),
        "blocker_reason": "" if consistency_ok else "baseline_and_optimized_path_diverged",
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage71_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage71_report.md").write_text(
        "\n".join(
            [
                "# Stage-71 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- candidate_count_measured: `{summary['candidate_count_measured']}`",
                f"- baseline_runtime_seconds: `{summary['baseline_runtime_seconds']}`",
                f"- optimized_runtime_seconds: `{summary['optimized_runtime_seconds']}`",
                f"- speedup_pct: `{summary['speedup_pct']}`",
                f"- meets_target_40pct: `{summary['meets_target_40pct']}`",
                f"- cache_key: `{summary['cache_key']}`",
                f"- consistency_delta: `{summary['consistency_delta']}`",
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
