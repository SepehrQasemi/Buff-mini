"""Run Stage-56 self-learning v4 using Stage-52 and Stage-53 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage56 import assess_learning_depth_v4, derive_allocation_adjustments, expand_registry_rows_v4
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-56 self-learning v4")
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


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage55 = _load_json(docs_dir / "stage55_summary.json")
    if str(stage55.get("stage28_run_id", "")).strip():
        return str(stage55["stage28_run_id"]).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    stage53_summary = _load_json(docs_dir / "stage53_summary.json")

    candidates = pd.DataFrame()
    predictions = pd.DataFrame()
    if stage28_run_id:
        cpath = Path(args.runs_dir) / stage28_run_id / "stage52" / "setup_candidates_v2.csv"
        ppath = Path(args.runs_dir) / stage28_run_id / "stage53" / "predictions.csv"
        if cpath.exists():
            candidates = pd.read_csv(cpath)
        if ppath.exists():
            predictions = pd.read_csv(ppath)

    input_mode = "stage52_stage53_artifacts"
    if candidates.empty or predictions.empty:
        input_mode = "bootstrap_registry_memory"
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": f"s56_{idx}",
                    "family": family,
                    "timeframe": timeframe,
                    "context": {"primary": "trend" if idx % 2 == 0 else "range"},
                    "trigger": {"type": "pullback" if idx % 2 == 0 else "sweep"},
                    "entry_logic": "enter_on_signal_close",
                    "stop_logic": "swing_fail",
                    "target_logic": "rr_projection",
                    "rr_model": {"first_target_rr": 1.2 + idx * 0.15},
                    "pre_replay_reject_reason": "REJECT::WEAK_TRIGGER" if idx % 3 == 0 else "",
                    "exp_lcb_proxy": -0.001 + idx * 0.002,
                }
                for idx, (family, timeframe) in enumerate(
                    [
                        ("structure_pullback_continuation", "15m"),
                        ("liquidity_sweep_reversal", "30m"),
                        ("squeeze_flow_breakout", "1h"),
                        ("structure_pullback_continuation", "2h"),
                        ("liquidity_sweep_reversal", "4h"),
                    ]
                )
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "candidate_id": f"s56_{idx}",
                    "expected_net_after_cost": -0.0003 + idx * 0.0009,
                    "mae_pct": -0.004 + idx * 0.0005,
                    "mfe_pct": 0.006 + idx * 0.0010,
                    "replay_priority": 0.45 + idx * 0.10,
                }
                for idx in range(5)
            ]
        )

    if "candidate_id" not in candidates.columns:
        candidates["candidate_id"] = [f"s56_auto_{idx}" for idx in range(len(candidates))]
    if "candidate_id" not in predictions.columns:
        predictions["candidate_id"] = [f"s56_auto_{idx}" for idx in range(len(predictions))]

    registry_rows = expand_registry_rows_v4(
        candidates=candidates,
        predictions=predictions,
        seed=int(cfg.get("search", {}).get("seed", 42)),
        run_id=stage28_run_id or "stage56_bootstrap",
    )
    adjustments = derive_allocation_adjustments(registry_rows)
    depth = assess_learning_depth_v4(registry_rows)
    dead_path = (
        int(stage53_summary.get("stage_a_survivors", 0)) <= 0
        or int(stage53_summary.get("stage_b_survivors", 0)) <= 0
        or not bool(stage53_summary.get("quality_gate_passed", False))
    )
    if dead_path:
        depth = "EARLY_OR_DEAD_PATH"
    summary = {
        "stage": "56",
        "status": "SUCCESS" if registry_rows and not dead_path else "PARTIAL",
        "input_mode": input_mode,
        "stage28_run_id": stage28_run_id,
        "registry_rows": len(registry_rows),
        "learning_depth": depth,
        "allocation_adjustments": adjustments,
        "blocker_reason": "dead_upstream_path" if dead_path else "",
        "summary_hash": stable_hash(
            {
                "status": "SUCCESS" if registry_rows and not dead_path else "PARTIAL",
                "input_mode": input_mode,
                "stage28_run_id": stage28_run_id,
                "registry_rows": len(registry_rows),
                "learning_depth": depth,
                "allocation_adjustments": adjustments,
                "blocker_reason": "dead_upstream_path" if dead_path else "",
            },
            length=16,
        ),
    }

    if stage28_run_id:
        out_dir = Path(args.runs_dir) / stage28_run_id / "stage56"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "learning_registry_v4.json").write_text(json.dumps(registry_rows, indent=2, allow_nan=False), encoding="utf-8")
        (out_dir / "allocation_adjustments.json").write_text(json.dumps(adjustments, indent=2, allow_nan=False), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage56_summary.json"
    report_path = docs_dir / "stage56_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-56 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- input_mode: `{summary['input_mode']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- registry_rows: `{summary['registry_rows']}`",
                f"- learning_depth: `{summary['learning_depth']}`",
                f"- allocation_adjustments: `{summary['allocation_adjustments']}`",
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
