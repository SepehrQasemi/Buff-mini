"""Helpers for bounded serious edge campaigns and scope evaluation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from buffmini.research.diagnostics import (
    classify_candidate_hierarchy,
    compute_near_miss_distance,
    resolve_first_death_stage,
)
from buffmini.research.modes import build_mode_context
from buffmini.research.robustness import evaluate_split_perturbation, summarize_layered_robustness
from buffmini.research.transfer import classify_transfer_outcome, discover_transfer_symbols
from buffmini.stage48.tradability_learning import Stage48Config, compute_stage48_labels, score_candidates_with_ranker
from buffmini.stage70 import generate_expanded_candidates
from buffmini.validation import (
    compute_transfer_metrics,
    estimate_trade_monte_carlo,
    evaluate_candidate_walkforward,
    evaluate_cross_perturbation,
    load_candidate_market_frame,
    run_candidate_replay,
)


DEFAULT_CAMPAIGN_FAMILIES = (
    "structure_pullback_continuation",
    "failed_breakout_reversal",
    "volatility_regime_transition",
    "exhaustion_mean_reversion",
    "liquidity_sweep_reversal",
    "squeeze_flow_breakout",
)


def select_campaign_families(feedback: dict[str, Any] | None = None, *, limit: int = 4) -> list[str]:
    adjustments = dict((feedback or {}).get("family_priority_adjustments", {}))
    ordered = list(DEFAULT_CAMPAIGN_FAMILIES)
    if adjustments:
        ordered = sorted(
            DEFAULT_CAMPAIGN_FAMILIES,
            key=lambda family: (-float(adjustments.get(family, 0.0)), DEFAULT_CAMPAIGN_FAMILIES.index(family)),
        )
    return [str(family) for family in ordered[: max(1, int(limit))]]


def classify_campaign_outcome(
    *,
    edge_inventory: list[dict[str, Any]],
    evaluated_assets: int,
    blocked_assets: int,
) -> str:
    robust = sum(1 for row in edge_inventory if str(row.get("final_class", "")) == "robust_candidate")
    promising = sum(1 for row in edge_inventory if str(row.get("final_class", "")) == "promising_but_unproven")
    if evaluated_assets <= 0 and blocked_assets > 0:
        return "system_blocked_uninterpretable"
    if robust > 0:
        return "robust_edge_candidates_present"
    if promising > 0:
        return "weak_promising_signs_need_refinement"
    return "honest_no_robust_edge_survived"


def evaluate_scope_campaign(
    *,
    config: dict[str, Any],
    symbol: str,
    timeframe: str,
    families: list[str],
    candidate_limit: int = 6,
    requested_mode: str = "exploration",
    auto_pin_resolved_end: bool = False,
    relax_continuity: bool = False,
    evaluate_transfer: bool = True,
    ranking_profile: str | None = None,
    data_source_override: str | None = None,
) -> dict[str, Any]:
    effective_cfg, mode_summary = build_mode_context(
        config,
        requested_mode=requested_mode,
        auto_pin_resolved_end=auto_pin_resolved_end,
    )
    if str(data_source_override or "").strip():
        effective_cfg = deepcopy(effective_cfg)
        effective_cfg.setdefault("research_run", {})["data_source"] = str(data_source_override).strip()
    if relax_continuity:
        effective_cfg = deepcopy(effective_cfg)
        effective_cfg.setdefault("data", {}).setdefault("continuity", {})["strict_mode"] = False
        effective_cfg["data"]["continuity"]["fail_on_gap"] = False
        effective_cfg.setdefault("reproducibility", {})["frozen_research_mode"] = False
        effective_cfg["reproducibility"]["require_resolved_end_ts"] = False

    frame, market_meta = load_candidate_market_frame(effective_cfg, symbol=symbol, timeframe=timeframe)
    blocked_reason = ""
    if frame.empty:
        blocked_reason = "missing_market_frame"
    elif bool(market_meta.get("runtime_truth_blocked", False)):
        blocked_reason = str(market_meta.get("runtime_truth_reason", "runtime_truth_blocked"))
    elif bool(market_meta.get("continuity_blocked", False)):
        blocked_reason = str(market_meta.get("continuity_reason", "continuity_blocked")) or "continuity_blocked"

    if blocked_reason:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "mode_summary": mode_summary,
            "market_meta": market_meta,
            "blocked": True,
            "blocked_reason": blocked_reason,
            "candidate_count": 0,
            "promising_count": 0,
            "validated_count": 0,
            "robust_count": 0,
            "blocked_count": 1,
            "dominant_failure_reasons": {blocked_reason: 1},
            "evaluations": [],
            "ranked_frame": pd.DataFrame(),
        }

    candidates = generate_expanded_candidates(
        discovery_timeframes=[str(timeframe)],
        budget_mode_selected=str((effective_cfg.get("budget_mode", {}) or {}).get("selected", "search")),
        active_families=list(families),
        min_search_candidates=400,
    )
    ranking_lookback_bars = int(max(256, ((effective_cfg.get("research_run", {}) or {}).get("ranking_lookback_bars", 4096) or 4096)))
    ranking_frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy().tail(ranking_lookback_bars).reset_index(drop=True)
    labels = compute_stage48_labels(
        ranking_frame.copy(),
        cfg=Stage48Config(
            round_trip_cost_pct=float((effective_cfg.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0,
        ),
    )
    ranked = score_candidates_with_ranker(
        candidates,
        labels,
        market_frame=ranking_frame.copy(),
        profile=str(ranking_profile or (effective_cfg.get("research_run", {}) or {}).get("ranking_profile", "stage99_quality_acceleration")),
    )
    merged = candidates.merge(ranked, on="candidate_id", how="inner")
    hierarchy_order = {
        "robust_candidate": 0,
        "validated_candidate": 1,
        "promising_but_unproven": 2,
        "interesting_but_fragile": 3,
        "junk": 4,
    }
    merged["_hierarchy_rank"] = merged.get("candidate_hierarchy", "junk").astype(str).map(lambda value: hierarchy_order.get(str(value), 9)).fillna(9)
    merged = merged.sort_values(["_hierarchy_rank", "rank_score", "candidate_id"], ascending=[True, False, True]).reset_index(drop=True)
    top = merged.head(max(1, int(candidate_limit))).copy()

    symbols = discover_transfer_symbols(effective_cfg, primary_symbol=symbol)
    other_symbols = [item for item in symbols if item != symbol]
    evaluations = evaluate_candidate_batch(
        candidates=top.to_dict(orient="records"),
        config=effective_cfg,
        symbol=symbol,
        frame=frame,
        market_meta=market_meta,
        transfer_symbol=other_symbols[0] if other_symbols else "",
        evaluate_transfer=bool(evaluate_transfer),
    )
    failure_counts: dict[str, int] = {}
    for row in evaluations:
        reason = str(row.get("death_reason", "")).strip()
        if reason:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    class_counts = pd.Series([str(row.get("final_class", "rejected")) for row in evaluations], dtype=str).value_counts().to_dict() if evaluations else {}
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "mode_summary": mode_summary,
        "market_meta": market_meta,
        "blocked": False,
        "blocked_reason": "",
        "ranking_lookback_bars_effective": ranking_lookback_bars,
        "candidate_count": int(len(candidates)),
        "promising_count": int(class_counts.get("promising_but_unproven", 0)),
        "validated_count": int(sum(1 for row in evaluations if str(row.get("candidate_hierarchy", "")) == "validated_candidate")),
        "robust_count": int(class_counts.get("robust_candidate", 0)),
        "blocked_count": 0,
        "dominant_failure_reasons": dict(sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[:5]),
        "evaluations": evaluations,
        "ranked_frame": merged.drop(columns=["_hierarchy_rank"], errors="ignore"),
    }


def evaluate_candidate_batch(
    *,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    symbol: str,
    frame: pd.DataFrame,
    market_meta: dict[str, Any],
    transfer_symbol: str = "",
    evaluate_transfer: bool = True,
) -> list[dict[str, Any]]:
    return [
        evaluate_candidate_record(
            candidate=dict(row),
            config=config,
            symbol=symbol,
            frame=frame,
            market_meta=market_meta,
            transfer_symbol=transfer_symbol,
            evaluate_transfer=bool(evaluate_transfer),
        )
        for row in list(candidates)
    ]


def evaluate_candidate_record(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
    frame: pd.DataFrame,
    market_meta: dict[str, Any],
    transfer_symbol: str = "",
    evaluate_transfer: bool = True,
    candidate_overrides: dict[str, Any] | None = None,
    replay_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_candidate = _merge_candidate_overrides(candidate, candidate_overrides)
    replay = run_candidate_replay(
        candidate=effective_candidate,
        config=config,
        symbol=symbol,
        frame=frame,
        market_meta=market_meta,
        **_clean_replay_options(replay_options),
    )
    replay_metrics = dict(replay.get("metrics", {}))
    transfer_diag = {"classification": "not_transferable", "diagnostics": ["no_secondary_asset"]}
    if not bool(evaluate_transfer):
        transfer_diag = {"classification": "not_evaluated", "diagnostics": ["transfer_skipped"]}
    elif str(transfer_symbol).strip():
        transfer_metrics = compute_transfer_metrics(candidate=effective_candidate, config=config, symbol=str(transfer_symbol))
        transfer_diag = classify_transfer_outcome(
            primary_metrics=replay_metrics,
            transfer_metrics=transfer_metrics,
        )
    if _replay_gate_failed(
        replay_trade_count=int(replay_metrics.get("trade_count", 0)),
        replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
        replay_allowed=bool(replay.get("decision_use_allowed", False)),
        config=config,
    ):
        walkforward_summary: dict[str, Any] = {}
        layered = {
            "level_reached": 0,
            "level_name": "mechanism_not_viable",
            "stop_reason": "replay_gate_failed",
            "stress_results": {
                "monte_carlo_passed": False,
                "cross_perturbation_passed": False,
                "split_perturbation_passed": False,
            },
        }
        death_stage, death_reason = resolve_first_death_stage(
            replay_trade_count=int(replay_metrics.get("trade_count", 0)),
            replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
            replay_allowed=bool(replay.get("decision_use_allowed", False)),
            walkforward_usable_windows=0,
            monte_carlo_passed=False,
            perturbation_passed=False,
            split_passed=False,
            transfer_classification=str(transfer_diag.get("classification", "")),
            market_blocked_reason="",
            config=config,
        )
        hierarchy = classify_candidate_hierarchy(
            rank_score=float(candidate.get("rank_score", 0.0)),
            replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
            walkforward_usable_windows=0,
            decision_use_allowed=bool(replay.get("decision_use_allowed", False)),
            aggregate_risk=float(candidate.get("aggregate_risk", 1.0)),
            robustness_level=0,
        )
        near_miss_distance = compute_near_miss_distance(
            replay_trade_count=int(replay_metrics.get("trade_count", 0)),
            replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
            walkforward_usable_windows=0,
            robustness_level=0,
            transfer_classification=str(transfer_diag.get("classification", "")),
            config=config,
        )
        return _finalize_candidate_record(
            candidate=effective_candidate,
            replay_metrics=replay_metrics,
            walkforward_summary=walkforward_summary,
            layered=layered,
            transfer_diag=transfer_diag,
            death_stage=death_stage,
            death_reason=death_reason,
            hierarchy=hierarchy,
            near_miss_distance=near_miss_distance,
        )
    walkforward = evaluate_candidate_walkforward(candidate=effective_candidate, config=config, symbol=symbol, frame=frame, market_meta=market_meta)
    monte_carlo = estimate_trade_monte_carlo(
        walkforward.get("forward_trades", pd.DataFrame()),
        seed=int((config.get("search", {}) or {}).get("seed", 42)),
        n_paths=500,
        block_size=8,
    )
    perturb = evaluate_cross_perturbation(candidate=effective_candidate, config=config, symbol=symbol, frame=frame, market_meta=market_meta)
    split = evaluate_split_perturbation(candidate=effective_candidate, config=config, symbol=symbol, frame=frame, market_meta=market_meta)
    layered = summarize_layered_robustness(
        replay_metrics=replay_metrics,
        walkforward_summary=dict(walkforward.get("summary", {})),
        monte_carlo=monte_carlo,
        perturbation=perturb,
        split_perturbation=split,
        config=config,
    )
    walkforward_summary = dict(walkforward.get("summary", {}))
    death_stage, death_reason = resolve_first_death_stage(
        replay_trade_count=int(replay_metrics.get("trade_count", 0)),
        replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
        replay_allowed=bool(replay.get("decision_use_allowed", False)),
        walkforward_usable_windows=int(walkforward_summary.get("usable_windows", 0)),
        monte_carlo_passed=bool(layered.get("stress_results", {}).get("monte_carlo_passed", False)),
        perturbation_passed=bool(layered.get("stress_results", {}).get("cross_perturbation_passed", False)),
        split_passed=bool(layered.get("stress_results", {}).get("split_perturbation_passed", False)),
        transfer_classification=str(transfer_diag.get("classification", "")),
        market_blocked_reason="",
        config=config,
    )
    hierarchy = classify_candidate_hierarchy(
        rank_score=float(candidate.get("rank_score", 0.0)),
        replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
        walkforward_usable_windows=int(walkforward_summary.get("usable_windows", 0)),
        decision_use_allowed=bool(replay.get("decision_use_allowed", False)),
        aggregate_risk=float(candidate.get("aggregate_risk", 1.0)),
        robustness_level=int(layered.get("level_reached", 0)),
    )
    near_miss_distance = compute_near_miss_distance(
        replay_trade_count=int(replay_metrics.get("trade_count", 0)),
        replay_exp_lcb=float(replay_metrics.get("exp_lcb", -1.0)),
        walkforward_usable_windows=int(walkforward_summary.get("usable_windows", 0)),
        robustness_level=int(layered.get("level_reached", 0)),
        transfer_classification=str(transfer_diag.get("classification", "")),
        config=config,
    )
    return _finalize_candidate_record(
        candidate=effective_candidate,
        replay_metrics=replay_metrics,
        walkforward_summary=walkforward_summary,
        layered=layered,
        transfer_diag=transfer_diag,
        death_stage=death_stage,
        death_reason=death_reason,
        hierarchy=hierarchy,
        near_miss_distance=near_miss_distance,
    )


def _replay_gate_failed(
    *,
    replay_trade_count: int,
    replay_exp_lcb: float,
    replay_allowed: bool,
    config: dict[str, Any],
) -> bool:
    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    min_trades = int(replay_gate.get("min_trade_count", 40))
    min_exp_lcb = float(replay_gate.get("min_exp_lcb", 0.0))
    return (not replay_allowed) or int(replay_trade_count) < int(min_trades) or float(replay_exp_lcb) < float(min_exp_lcb)


def _merge_candidate_overrides(
    candidate: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(candidate)
    payload = deepcopy(overrides or {})
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged.get(key, {}))
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _clean_replay_options(replay_options: dict[str, Any] | None) -> dict[str, Any]:
    allowed = {
        "cost_multiplier",
        "slippage_multiplier",
        "funding_multiplier",
        "hold_bars_multiplier",
        "signal_delay_bars",
        "signal_keep_ratio",
    }
    options = {}
    for key, value in dict(replay_options or {}).items():
        if key in allowed:
            options[key] = value
    return options


def _finalize_candidate_record(
    *,
    candidate: dict[str, Any],
    replay_metrics: dict[str, Any],
    walkforward_summary: dict[str, Any],
    layered: dict[str, Any],
    transfer_diag: dict[str, Any],
    death_stage: str,
    death_reason: str,
    hierarchy: str,
    near_miss_distance: float,
) -> dict[str, Any]:
    if hierarchy == "robust_candidate" and str(transfer_diag.get("classification", "")) in {"transferable", "partially_transferable"}:
        final_class = "robust_candidate"
    elif hierarchy in {"validated_candidate", "promising_but_unproven", "interesting_but_fragile"}:
        final_class = "promising_but_unproven"
    else:
        final_class = "rejected"
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "family": str(candidate.get("family", "")),
        "subfamily": str(candidate.get("subfamily", "")),
        "expected_regime": str(candidate.get("expected_regime", "")),
        "rank_score": float(candidate.get("rank_score", 0.0)),
        "candidate_class": str(candidate.get("candidate_class", "")),
        "candidate_hierarchy": hierarchy,
        "usefulness_prior": float(candidate.get("usefulness_prior", 0.0)),
        "trade_quality_bonus": float(candidate.get("trade_quality_bonus", 0.0)),
        "aggregate_risk": float(candidate.get("aggregate_risk", 1.0)),
        "trade_density_risk": float(candidate.get("trade_density_risk", 1.0)),
        "cost_fragility_risk": float(candidate.get("cost_fragility_risk", 1.0)),
        "regime_concentration_risk": float(candidate.get("regime_concentration_risk", 1.0)),
        "hold_sanity_risk": float(candidate.get("hold_sanity_risk", 1.0)),
        "overlap_duplication_risk": float(candidate.get("overlap_duplication_risk", 1.0)),
        "clustering_risk": float(candidate.get("clustering_risk", 1.0)),
        "thin_evidence_risk": float(candidate.get("thin_evidence_risk", 1.0)),
        "transfer_risk_prior": float(candidate.get("transfer_risk_prior", 1.0)),
        "replay_trade_count": int(replay_metrics.get("trade_count", 0)),
        "replay_exp_lcb": float(replay_metrics.get("exp_lcb", -1.0)),
        "walkforward_usable_windows": int(walkforward_summary.get("usable_windows", 0)),
        "robustness_level": int(layered.get("level_reached", 0)),
        "robustness_level_name": str(layered.get("level_name", "")),
        "robustness_stop_reason": str(layered.get("stop_reason", "")),
        "monte_carlo_passed": bool(layered.get("stress_results", {}).get("monte_carlo_passed", False)),
        "cross_perturbation_passed": bool(layered.get("stress_results", {}).get("cross_perturbation_passed", False)),
        "split_perturbation_passed": bool(layered.get("stress_results", {}).get("split_perturbation_passed", False)),
        "transfer_classification": str(transfer_diag.get("classification", "")),
        "transfer_diagnostics": list(transfer_diag.get("diagnostics", [])),
        "first_death_stage": death_stage,
        "death_reason": death_reason,
        "near_miss_distance": float(near_miss_distance),
        "final_class": final_class,
    }
