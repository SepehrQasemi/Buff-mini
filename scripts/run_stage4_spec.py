"""Generate Stage-4 trading specification documents."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT, RUNS_DIR
from buffmini.execution.simulator import resolve_stage4_method_and_leverage
from buffmini.spec.trading_spec import generate_trading_spec
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-4 spec generator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--stage3-3-run-id", type=str, default=None)
    parser.add_argument("--output-doc", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--export-global-docs", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage2_summary = _load_json(args.runs_dir / args.stage2_run_id / "portfolio_summary.json")

    stage3_summary = None
    if args.stage3_3_run_id:
        stage3_summary = _load_json(args.runs_dir / args.stage3_3_run_id / "selector_summary.json")

    selected_method, selected_leverage, from_stage3, warnings = resolve_stage4_method_and_leverage(
        cfg=config,
        stage3_choice=stage3_summary,
    )
    if warnings:
        for item in warnings:
            print(f"warning: {item}")
    if selected_method not in stage2_summary.get("portfolio_methods", {}):
        raise ValueError(f"Selected method `{selected_method}` not found in Stage-2 summary")

    method_payload = stage2_summary["portfolio_methods"][selected_method]
    weights = {str(candidate_id): float(weight) for candidate_id, weight in method_payload.get("weights", {}).items()}
    candidate_ids = [candidate_id for candidate_id, weight in weights.items() if float(weight) > 0]
    stage1_run_id = str(stage2_summary["stage1_run_id"])
    stage1_candidates_dir = args.runs_dir / stage1_run_id / "candidates"
    candidate_lookup = _load_candidate_lookup(stage1_candidates_dir)
    selected_candidates: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        payload = candidate_lookup.get(candidate_id)
        if payload is None:
            continue
        selected_candidates.append(
            {
                "candidate_id": candidate_id,
                "strategy_family": str(payload["strategy_family"]),
                "gating": str(payload["gating"]),
                "exit_mode": str(payload["exit_mode"]),
                "parameters": dict(payload["parameters"]),
                "weight": float(weights[candidate_id]),
            }
        )

    run_id_payload = {
        "stage2_run_id": args.stage2_run_id,
        "stage3_3_run_id": args.stage3_3_run_id,
        "config_hash": stable_hash(config, length=16),
        "selected_method": selected_method,
        "selected_leverage": float(selected_leverage),
        "candidate_ids": candidate_ids,
    }
    run_id = args.run_id or f"{utc_now_compact()}_{stable_hash(run_id_payload, length=12)}_stage4"
    run_dir = args.runs_dir / run_id
    spec_dir = run_dir / "spec"
    output_doc = Path(args.output_doc) if args.output_doc is not None else spec_dir / "trading_spec.md"

    outputs = generate_trading_spec(
        cfg=config,
        stage2_metadata=stage2_summary,
        stage3_3_choice=stage3_summary,
        selected_candidates=selected_candidates,
        output_path=output_doc,
    )

    policy_snapshot = _build_policy_snapshot(
        config=config,
        selected_method=selected_method,
        selected_leverage=float(selected_leverage),
    )

    run_summary = {
        "run_id": run_id,
        "stage2_run_id": args.stage2_run_id,
        "stage3_3_run_id": args.stage3_3_run_id,
        "stage1_run_id": stage1_run_id,
        "selected_method": selected_method,
        "selected_leverage": float(selected_leverage),
        "selected_from_stage3_3": bool(from_stage3),
        "selected_candidates": candidate_ids,
        "trading_spec_path": str(outputs["trading_spec_path"]),
        "paper_checklist_path": str(outputs["paper_checklist_path"]),
        "policy_snapshot_path": str(run_dir / "policy_snapshot.json"),
        "exported_global_docs": bool(args.export_global_docs),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    (run_dir / "policy_snapshot.json").write_text(json.dumps(policy_snapshot, indent=2), encoding="utf-8")

    if args.export_global_docs:
        global_spec = PROJECT_ROOT / "docs" / "trading_spec.md"
        global_checklist = PROJECT_ROOT / "docs" / "paper_trading_checklist.md"
        global_spec.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(outputs["trading_spec_path"], global_spec)
        shutil.copyfile(outputs["paper_checklist_path"], global_checklist)

    print(f"stage4_run_id: {run_id}")
    print(f"selected_method: {selected_method}")
    print(f"selected_leverage: {selected_leverage}x")
    print(f"source: {'Stage-3.3' if from_stage3 else 'Stage-4 defaults'}")
    print(f"trading_spec: {outputs['trading_spec_path']}")
    print(f"paper_trading_checklist: {outputs['paper_checklist_path']}")
    print(f"policy_snapshot: {run_dir / 'policy_snapshot.json'}")
    print(f"export_global_docs: {bool(args.export_global_docs)}")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_candidate_lookup(candidates_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in sorted(candidates_dir.glob("strategy_*.json")):
        payload = _load_json(path)
        lookup[str(payload["candidate_id"])] = payload
    return lookup


def _build_policy_snapshot(config: dict[str, Any], selected_method: str, selected_leverage: float) -> dict[str, Any]:
    risk_cfg = config["risk"]
    return {
        "selected_method": str(selected_method),
        "leverage": float(selected_leverage),
        "execution_mode": str(config["execution"]["mode"]),
        "caps": {
            "max_gross_exposure": float(risk_cfg["max_gross_exposure"]),
            "max_net_exposure_per_symbol": float(risk_cfg["max_net_exposure_per_symbol"]),
            "max_open_positions": int(risk_cfg["max_open_positions"]),
        },
        "costs": {
            "round_trip_cost_pct": float(config["costs"]["round_trip_cost_pct"]),
            "slippage_pct": float(config["costs"]["slippage_pct"]),
            "funding_pct_per_day": float(config["costs"]["funding_pct_per_day"]),
        },
        "kill_switch": {
            "enabled": bool(risk_cfg["killswitch"]["enabled"]),
            "max_daily_loss_pct": float(risk_cfg["killswitch"]["max_daily_loss_pct"]),
            "max_peak_to_valley_dd_pct": float(risk_cfg["killswitch"]["max_peak_to_valley_dd_pct"]),
            "max_consecutive_losses": int(risk_cfg["killswitch"]["max_consecutive_losses"]),
            "cool_down_bars": int(risk_cfg["killswitch"]["cool_down_bars"]),
        },
        "config_hash": stable_hash(config, length=16),
    }


if __name__ == "__main__":
    main()
