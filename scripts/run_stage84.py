"""Run Stage-84 first serious edge campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.campaign import classify_campaign_outcome, select_campaign_families
from buffmini.research.modes import build_mode_context
from buffmini.research.robustness import evaluate_split_perturbation, summarize_layered_robustness
from buffmini.research.transfer import classify_transfer_outcome, discover_transfer_symbols
from buffmini.stage48.tradability_learning import Stage48Config, compute_stage48_labels, score_candidates_with_ranker
from buffmini.stage70 import generate_expanded_candidates
from buffmini.utils.hashing import stable_hash
from buffmini.validation import (
    compute_transfer_metrics,
    estimate_trade_monte_carlo,
    evaluate_candidate_walkforward,
    evaluate_cross_perturbation,
    load_candidate_market_frame,
    run_candidate_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-84 first serious edge campaign")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--max-candidates-per-asset", type=int, default=4)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_json_mapping(raw: Any) -> dict[str, float]:
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config))
    effective_cfg, mode_summary = build_mode_context(cfg, requested_mode="evaluation", auto_pin_resolved_end=True)
    feedback = _load_json(docs_dir / "stage82_search_feedback.json")
    families = select_campaign_families(feedback, limit=4)
    symbols = discover_transfer_symbols(effective_cfg)[:2]
    edge_rows: list[dict[str, Any]] = []
    blocked_asset_details: list[dict[str, Any]] = []
    blocked_assets = 0
    evaluated_assets = 0

    for symbol in symbols:
        frame, market_meta = load_candidate_market_frame(effective_cfg, symbol=symbol, timeframe="1h")
        if frame.empty or bool(market_meta.get("runtime_truth_blocked", False)) or bool(market_meta.get("continuity_blocked", False)):
            blocked_assets += 1
            blocked_asset_details.append(
                {
                    "asset": symbol,
                    "runtime_truth_blocked": bool(market_meta.get("runtime_truth_blocked", False)),
                    "runtime_truth_reason": str(market_meta.get("runtime_truth_reason", "")),
                    "continuity_blocked": bool(market_meta.get("continuity_blocked", False)),
                    "continuity_reason": str(market_meta.get("continuity_reason", "")),
                    "resolved_end_ts": str(market_meta.get("resolved_end_ts", "")),
                    "data_hash": str(market_meta.get("data_hash", "")),
                    "row_count": int(market_meta.get("row_count", 0)),
                }
            )
            continue
        evaluated_assets += 1
        candidates = generate_expanded_candidates(
            discovery_timeframes=["1h"],
            budget_mode_selected="search",
            active_families=families,
            failure_feedback=feedback,
            min_search_candidates=400,
        )
        labels = compute_stage48_labels(
            frame[["timestamp", "open", "high", "low", "close", "volume"]].copy(),
            cfg=Stage48Config(round_trip_cost_pct=float((effective_cfg.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0),
        )
        ranked = score_candidates_with_ranker(candidates, labels, market_frame=frame[["timestamp", "open", "high", "low", "close", "volume"]].copy())
        merged = candidates.merge(ranked, on="candidate_id", how="inner").sort_values(["candidate_class", "rank_score"], ascending=[True, False]).reset_index(drop=True)
        top = merged.head(max(1, int(args.max_candidates_per_asset)))
        for row in top.to_dict(orient="records"):
            replay = run_candidate_replay(candidate=row, config=effective_cfg, symbol=symbol, frame=frame, market_meta=market_meta)
            walkforward = evaluate_candidate_walkforward(candidate=row, config=effective_cfg, symbol=symbol, frame=frame, market_meta=market_meta)
            monte_carlo = estimate_trade_monte_carlo(
                walkforward.get("forward_trades", pd.DataFrame()),
                seed=int((effective_cfg.get("search", {}) or {}).get("seed", 42)),
                n_paths=500,
                block_size=8,
            )
            perturb = evaluate_cross_perturbation(candidate=row, config=effective_cfg, symbol=symbol, frame=frame, market_meta=market_meta)
            split = evaluate_split_perturbation(candidate=row, config=effective_cfg, symbol=symbol, frame=frame, market_meta=market_meta)
            layered = summarize_layered_robustness(
                replay_metrics=dict(replay.get("metrics", {})),
                walkforward_summary=dict(walkforward.get("summary", {})),
                monte_carlo=monte_carlo,
                perturbation=perturb,
                split_perturbation=split,
                config=effective_cfg,
            )
            other_symbols = [item for item in symbols if item != symbol]
            transfer_diag = {"classification": "not_transferable", "diagnostics": ["no_secondary_asset"]}
            if other_symbols:
                transfer_metrics = compute_transfer_metrics(candidate=row, config=effective_cfg, symbol=other_symbols[0])
                transfer_diag = classify_transfer_outcome(
                    primary_metrics=dict(replay.get("metrics", {})),
                    transfer_metrics=transfer_metrics,
                )
            if int(layered.get("level_reached", 0)) >= 3 and str(transfer_diag.get("classification", "")) in {"transferable", "partially_transferable"}:
                final_class = "robust_candidate"
            elif str(row.get("candidate_class", "")) == "promising_but_unproven" or int(layered.get("level_reached", 0)) >= 1:
                final_class = "promising_but_unproven"
            else:
                final_class = "rejected"
            regime_map = _parse_json_mapping(row.get("regime_activation_map", "{}"))
            dominant_regime = max(regime_map, key=regime_map.get) if regime_map else "unknown"
            edge_rows.append(
                {
                    "asset": symbol,
                    "candidate_id": str(row.get("candidate_id", "")),
                    "family": str(row.get("family", "")),
                    "rank_score": float(row.get("rank_score", 0.0)),
                    "candidate_class": str(row.get("candidate_class", "")),
                    "robustness_level": int(layered.get("level_reached", 0)),
                    "robustness_level_name": str(layered.get("level_name", "")),
                    "transfer_classification": str(transfer_diag.get("classification", "")),
                    "transfer_diagnostics": list(transfer_diag.get("diagnostics", [])),
                    "dominant_regime": dominant_regime,
                    "final_class": final_class,
                    "stop_reason": str(layered.get("stop_reason", "")),
                }
            )

    edge_inventory = pd.DataFrame(edge_rows)
    if not edge_inventory.empty:
        edge_inventory.to_csv(docs_dir / "stage84_edge_inventory.csv", index=False)
    mechanism_map = (
        edge_inventory.groupby(["family", "final_class"], dropna=False).size().reset_index(name="count").to_dict(orient="records")
        if not edge_inventory.empty
        else []
    )
    regime_map = (
        edge_inventory.groupby(["asset", "dominant_regime"], dropna=False).size().reset_index(name="count").to_dict(orient="records")
        if not edge_inventory.empty
        else []
    )
    failure_analysis: dict[str, int] = {}
    for row in edge_rows:
        reason = str(row.get("stop_reason", "")).strip()
        if reason:
            failure_analysis[reason] = failure_analysis.get(reason, 0) + 1
        for diag in list(row.get("transfer_diagnostics", [])):
            key = f"transfer::{diag}"
            failure_analysis[key] = failure_analysis.get(key, 0) + 1
    for row in blocked_asset_details:
        if bool(row.get("runtime_truth_blocked", False)):
            key = f"blocked_runtime::{str(row.get('runtime_truth_reason', '')).strip() or 'unknown'}"
            failure_analysis[key] = failure_analysis.get(key, 0) + 1
        if bool(row.get("continuity_blocked", False)):
            key = f"blocked_continuity::{str(row.get('continuity_reason', '')).strip() or 'unknown'}"
            failure_analysis[key] = failure_analysis.get(key, 0) + 1

    campaign_outcome = classify_campaign_outcome(edge_inventory=edge_rows, evaluated_assets=evaluated_assets, blocked_assets=blocked_assets)
    robust_candidates = [row for row in edge_rows if str(row.get("final_class", "")) == "robust_candidate"]
    promising_candidates = [row for row in edge_rows if str(row.get("final_class", "")) == "promising_but_unproven"]
    status = "SUCCESS" if evaluated_assets > 0 else "PARTIAL"
    summary = {
        "stage": "84",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "SERIOUS_EDGE_CAMPAIGN_COMPLETE" if evaluated_assets > 0 else "SERIOUS_EDGE_CAMPAIGN_BLOCKED",
        "mode": str(mode_summary.get("mode", "")),
        "interpretation_allowed": bool(mode_summary.get("interpretation_allowed", False)),
        "baseline_resolved_end_ts": str(mode_summary.get("resolved_end_ts", "")),
        "assets": symbols,
        "mechanism_families": families,
        "evaluated_assets": int(evaluated_assets),
        "blocked_assets": int(blocked_assets),
        "edge_inventory_count": int(len(edge_rows)),
        "mechanism_map": mechanism_map,
        "regime_map": regime_map,
        "blocked_asset_details": blocked_asset_details,
        "candidate_class_counts": {
            "robust_candidate": int(len(robust_candidates)),
            "promising_but_unproven": int(len(promising_candidates)),
            "rejected": int(sum(1 for row in edge_rows if str(row.get("final_class", "")) == "rejected")),
        },
        "failure_analysis": failure_analysis,
        "promising_candidates": promising_candidates,
        "robust_candidates": robust_candidates,
        "campaign_outcome": campaign_outcome,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage84_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-84 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- mode: `{summary['mode']}`",
        f"- interpretation_allowed: `{summary['interpretation_allowed']}`",
        f"- baseline_resolved_end_ts: `{summary['baseline_resolved_end_ts']}`",
        f"- assets: `{summary['assets']}`",
        f"- mechanism_families: `{summary['mechanism_families']}`",
        f"- evaluated_assets: `{summary['evaluated_assets']}`",
        f"- blocked_assets: `{summary['blocked_assets']}`",
        f"- edge_inventory_count: `{summary['edge_inventory_count']}`",
        f"- campaign_outcome: `{summary['campaign_outcome']}`",
        "",
        "## Blocked Assets",
    ]
    for row in blocked_asset_details:
        lines.append(
            "- "
            f"{row['asset']}: runtime_truth_blocked=`{row['runtime_truth_blocked']}` "
            f"continuity_blocked=`{row['continuity_blocked']}` "
            f"runtime_truth_reason=`{row['runtime_truth_reason']}` "
            f"continuity_reason=`{row['continuity_reason']}`"
        )
    lines.extend([
        "",
        "## Candidate Classes",
    ])
    for key, value in sorted(summary["candidate_class_counts"].items()):
        lines.append(f"- {key}: `{int(value)}`")
    lines.extend(["", "## Failure Analysis"])
    for key, value in sorted(failure_analysis.items()):
        lines.append(f"- {key}: `{int(value)}`")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    (docs_dir / "stage84_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
