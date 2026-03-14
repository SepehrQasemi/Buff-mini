"""Run Stage-54 timeframe discovery and search optimizer using Stage-53 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage51 import resolve_research_scope
from buffmini.stage54 import build_timeframe_metrics, hyperband_prune, select_timeframe_promotions, tpe_suggest
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-54 timeframe discovery")
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
    stage53 = _load_json(docs_dir / "stage53_summary.json")
    if str(stage53.get("stage28_run_id", "")).strip():
        return str(stage53["stage28_run_id"]).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    scope = resolve_research_scope(cfg)
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
        input_mode = "bootstrap_timeframe_scan"
        rows: list[dict[str, Any]] = []
        for tf_idx, timeframe in enumerate(scope["discovery_timeframes"]):
            for fam_idx, family in enumerate(scope["active_setup_families"]):
                rows.append(
                    {
                        "candidate_id": f"s54_{family}_{timeframe}",
                        "family": str(family),
                        "timeframe": str(timeframe),
                        "rr_model": {"first_target_rr": round(1.35 + fam_idx * 0.20 + tf_idx * 0.06, 6)},
                        "cost_edge_proxy": round(-0.0002 + fam_idx * 0.001 + tf_idx * 0.0004, 8),
                        "tp_before_sl_prob": round(0.48 + fam_idx * 0.08 + tf_idx * 0.03, 6),
                        "expected_net_after_cost": round(-0.0004 + fam_idx * 0.0012 + tf_idx * 0.0003, 8),
                        "replay_priority": round(0.40 + fam_idx * 0.12 + tf_idx * 0.04, 6),
                        "pre_replay_reject_reason": "" if (fam_idx + tf_idx) % 3 else "REJECT::COST_MARGIN_TOO_LOW",
                    }
                )
        merged = pd.DataFrame(rows)
    else:
        merged = candidates.merge(predictions, on="candidate_id", how="left")
        if "timeframe" not in merged.columns:
            merged["timeframe"] = "1h"
        if "rr_model" not in merged.columns:
            merged["rr_model"] = [{"first_target_rr": 1.5}] * len(merged)
        if "pre_replay_reject_reason" not in merged.columns:
            merged["pre_replay_reject_reason"] = ""
        if "tp_before_sl_prob" not in merged.columns:
            merged["tp_before_sl_prob"] = 0.0
        if "expected_net_after_cost" not in merged.columns:
            merged["expected_net_after_cost"] = 0.0
        if "replay_priority" not in merged.columns:
            merged["replay_priority"] = 0.0

    stage43 = _load_json(docs_dir / "stage43_performance_summary.json")
    replay_runtime = float(stage43.get("phase_runtime_seconds", {}).get("replay_backtest", 0.0))
    runtime_by_timeframe = {
        str(timeframe): float((replay_runtime / max(1, len(scope["discovery_timeframes"]))) / max(1.0, float(idx + 1)))
        for idx, timeframe in enumerate(scope["discovery_timeframes"])
    }
    metrics = build_timeframe_metrics(merged, runtime_by_timeframe=runtime_by_timeframe)
    promotion = select_timeframe_promotions(
        metrics,
        promotion_timeframes=int(scope["promotion_timeframes"]),
        final_validation_timeframes=int(scope["final_validation_timeframes"]),
    )
    keep_n = int(cfg.get("budget_mode", {}).get("search", {}).get("stage_a_limit", 6))
    pruned = hyperband_prune(merged, keep=keep_n)

    history_path = Path(args.runs_dir) / stage28_run_id / "stage54" / "optimizer_history.csv" if stage28_run_id else None
    if history_path is not None and history_path.exists():
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame(
            [
                {"threshold": 0.50, "weight": 0.10, "objective": 0.01},
                {"threshold": 0.55, "weight": 0.15, "objective": 0.03},
                {"threshold": 0.60, "weight": 0.20, "objective": 0.02},
            ]
        )
    suggestion = tpe_suggest(history, search_space={"threshold": [0.50, 0.55, 0.60], "weight": [0.10, 0.15, 0.20]})

    stage_a_survivors = int(stage53_summary.get("stage_a_survivors", 0))
    stage_b_survivors = int(stage53_summary.get("stage_b_survivors", 0))
    dead_path = stage_a_survivors <= 0 or stage_b_survivors <= 0
    if dead_path:
        promotion = {"promotion_timeframes": [], "final_validation_timeframes": []}
        if "hyperband_keep" in pruned.columns:
            pruned["hyperband_keep"] = False

    summary = {
        "stage": "54",
        "status": "PARTIAL" if dead_path else "SUCCESS",
        "input_mode": input_mode,
        "stage28_run_id": stage28_run_id,
        "promotion_timeframes": promotion["promotion_timeframes"],
        "final_validation_timeframes": promotion["final_validation_timeframes"],
        "pruned_candidate_count": int(pruned.loc[pruned["hyperband_keep"], :].shape[0]) if "hyperband_keep" in pruned.columns else 0,
        "tpe_suggestion": suggestion,
        "blocker_reason": "dead_stage53_path" if dead_path else "",
        "summary_hash": stable_hash(
            {
                "status": "PARTIAL" if dead_path else "SUCCESS",
                "input_mode": input_mode,
                "stage28_run_id": stage28_run_id,
                "promotion_timeframes": promotion["promotion_timeframes"],
                "final_validation_timeframes": promotion["final_validation_timeframes"],
                "pruned_candidate_count": int(pruned.loc[pruned["hyperband_keep"], :].shape[0]) if "hyperband_keep" in pruned.columns else 0,
                "tpe_suggestion": suggestion,
                "blocker_reason": "dead_stage53_path" if dead_path else "",
            },
            length=16,
        ),
    }

    if stage28_run_id:
        out_dir = Path(args.runs_dir) / stage28_run_id / "stage54"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_dir / "timeframe_metrics.csv", index=False)
        pruned.to_csv(out_dir / "pruned_candidates.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage54_summary.json"
    report_path = docs_dir / "stage54_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-54 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- input_mode: `{summary['input_mode']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- promotion_timeframes: `{summary['promotion_timeframes']}`",
                f"- final_validation_timeframes: `{summary['final_validation_timeframes']}`",
                f"- pruned_candidate_count: `{summary['pruned_candidate_count']}`",
                f"- tpe_suggestion: `{summary['tpe_suggestion']}`",
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
