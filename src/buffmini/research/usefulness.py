"""Stage-95 live usefulness diagnostics and family replay-death mapping."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from buffmini.research.modes import build_mode_context
from buffmini.stage70.search_expansion import EXPANDED_FAMILIES
from buffmini.stage70 import generate_expanded_candidates
from buffmini.stage48.tradability_learning import Stage48Config, compute_stage48_labels, score_candidates_with_ranker
from buffmini.stage51.scope import resolve_research_scope
from buffmini.utils.hashing import stable_hash
from buffmini.validation import load_candidate_market_frame, run_candidate_replay


USEFUL_HIERARCHIES = {
    "interesting_but_fragile",
    "promising_but_unproven",
    "validated_candidate",
    "robust_candidate",
}


def evaluate_live_usefulness(
    config: dict[str, Any],
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    candidate_limit_per_family: int = 6,
    replay_window_bars: int = 2048,
) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    families = [str(value) for value in (scope.get("available_setup_families") or list(EXPANDED_FAMILIES))]
    effective_cfg, mode_summary = build_mode_context(
        config,
        requested_mode="exploration",
        auto_pin_resolved_end=False,
    )
    effective_cfg.setdefault("data", {}).setdefault("continuity", {})["strict_mode"] = False
    effective_cfg["data"]["continuity"]["fail_on_gap"] = False
    effective_cfg.setdefault("reproducibility", {})["frozen_research_mode"] = False
    effective_cfg["reproducibility"]["require_resolved_end_ts"] = False
    frame, market_meta = load_candidate_market_frame(effective_cfg, symbol=symbol, timeframe=timeframe)
    ranking_lookback_bars = int(max(256, ((effective_cfg.get("research_run", {}) or {}).get("ranking_lookback_bars", 4096) or 4096)))
    ranking_frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].copy().tail(ranking_lookback_bars).reset_index(drop=True)
    replay_frame = frame.copy().tail(max(256, int(replay_window_bars))).reset_index(drop=True)
    labels = compute_stage48_labels(
        ranking_frame.copy(),
        cfg=Stage48Config(
            round_trip_cost_pct=float((effective_cfg.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0,
        ),
    )
    before_rows: list[dict[str, Any]] = []
    after_rows: list[dict[str, Any]] = []
    family_reports: list[dict[str, Any]] = []
    candidates = generate_expanded_candidates(
        discovery_timeframes=[str(timeframe)],
        budget_mode_selected=str((effective_cfg.get("budget_mode", {}) or {}).get("selected", "search")),
        active_families=families,
        min_search_candidates=400,
    )
    ranked_before = _rank_candidates(
        candidates=candidates,
        labels=labels,
        ranking_frame=ranking_frame,
        ranking_profile="stage94_baseline",
    )
    ranked_after = _rank_candidates(
        candidates=candidates,
        labels=labels,
        ranking_frame=ranking_frame,
        ranking_profile="stage95_usefulness_push",
    )

    for family in families:
        before = _evaluate_ranked_usefulness(
            ranked_candidates=ranked_before,
            full_frame=replay_frame,
            market_meta=market_meta,
            config=effective_cfg,
            symbol=symbol,
            family=family,
            candidate_limit=int(candidate_limit_per_family),
        )
        after = _evaluate_ranked_usefulness(
            ranked_candidates=ranked_after,
            full_frame=replay_frame,
            market_meta=market_meta,
            config=effective_cfg,
            symbol=symbol,
            family=family,
            candidate_limit=int(candidate_limit_per_family),
        )
        before_rows.extend(list(before))
        after_rows.extend(list(after))
        family_reports.append(_family_usefulness_row(family=family, before=before, after=after))

    usefulness_delta = compare_usefulness(before_rows=before_rows, after_rows=after_rows)
    replay_death_map = build_family_replay_death_map(after_rows)
    dead_weight_families = identify_dead_weight_families(family_reports)
    frozen_families = {str(value) for value in (scope.get("frozen_setup_families") or [])}
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "families": families,
        "candidate_limit_per_family": int(candidate_limit_per_family),
        "replay_window_bars": int(max(256, int(replay_window_bars))),
        "before_profile": "stage94_baseline",
        "after_profile": "stage95_usefulness_push",
        "mode_summary": mode_summary,
        "before_counts": usefulness_delta["before_counts"],
        "after_counts": usefulness_delta["after_counts"],
        "usefulness_delta": usefulness_delta["delta"],
        "family_usefulness": family_reports,
        "family_replay_death_map": replay_death_map,
        "dead_weight_families": dead_weight_families,
        "stage95b_recommended": bool(dead_weight_families),
        "stage95b_applied": bool(any(str(row.get("family", "")) in frozen_families for row in dead_weight_families)),
        "summary_hash": stable_hash(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "usefulness_delta": usefulness_delta["delta"],
                "dead_weight_families": dead_weight_families,
            },
            length=16,
        ),
    }


def compare_usefulness(
    *,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    before_counts = _summarize_usefulness_counts(before_rows)
    after_counts = _summarize_usefulness_counts(after_rows)
    delta = {
        "useful_candidate_delta": int(after_counts["useful_candidate_count"] - before_counts["useful_candidate_count"]),
        "promising_delta": int(after_counts["promising_count"] - before_counts["promising_count"]),
        "mean_rank_score_delta": round(float(after_counts["mean_rank_score"] - before_counts["mean_rank_score"]), 6),
        "mean_near_miss_delta": round(float(after_counts["mean_near_miss_distance"] - before_counts["mean_near_miss_distance"]), 6),
        "replay_death_fraction_delta": round(float(after_counts["replay_death_fraction"] - before_counts["replay_death_fraction"]), 6),
    }
    return {
        "before_counts": before_counts,
        "after_counts": after_counts,
        "delta": delta,
    }


def build_family_replay_death_map(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row.get("family", ""))
        payload = grouped.setdefault(
            family,
            {
                "candidate_count": 0,
                "replay_deaths": 0,
                "walkforward_deaths": 0,
                "transfer_deaths": 0,
                "survivors": 0,
            },
        )
        payload["candidate_count"] += 1
        stage = str(row.get("first_death_stage", ""))
        if stage == "replay":
            payload["replay_deaths"] += 1
        elif stage == "walkforward":
            payload["walkforward_deaths"] += 1
        elif stage == "transfer":
            payload["transfer_deaths"] += 1
        elif stage == "survived":
            payload["survivors"] += 1
    result = []
    for family, counts in sorted(grouped.items()):
        candidate_count = max(1, int(counts["candidate_count"]))
        result.append(
            {
                "family": family,
                **counts,
                "replay_death_fraction": round(float(counts["replay_deaths"] / candidate_count), 6),
            }
        )
    return result


def identify_dead_weight_families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        useful_count = int(row.get("after_useful_candidate_count", 0))
        replay_fraction = float(row.get("after_replay_death_fraction", 0.0))
        near_miss_count = int(row.get("after_near_miss_count", 0))
        if useful_count == 0 and replay_fraction >= 0.75 and near_miss_count == 0:
            out.append(
                {
                    "family": str(row.get("family", "")),
                    "reason": "no_useful_survivors_and_replay_dominated",
                    "after_candidate_count": int(row.get("after_candidate_count", 0)),
                    "after_replay_death_fraction": round(replay_fraction, 6),
                }
            )
    return out


def _family_usefulness_row(*, family: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_counts = _summarize_usefulness_counts(list(before))
    after_counts = _summarize_usefulness_counts(list(after))
    return {
        "family": family,
        "before_candidate_count": int(before_counts["candidate_count"]),
        "before_useful_candidate_count": int(before_counts["useful_candidate_count"]),
        "before_promising_count": int(before_counts["promising_count"]),
        "before_replay_death_fraction": float(before_counts["replay_death_fraction"]),
        "before_near_miss_count": int(before_counts["near_miss_count"]),
        "after_candidate_count": int(after_counts["candidate_count"]),
        "after_useful_candidate_count": int(after_counts["useful_candidate_count"]),
        "after_promising_count": int(after_counts["promising_count"]),
        "after_replay_death_fraction": float(after_counts["replay_death_fraction"]),
        "after_near_miss_count": int(after_counts["near_miss_count"]),
        "useful_delta": int(after_counts["useful_candidate_count"] - before_counts["useful_candidate_count"]),
        "promising_delta": int(after_counts["promising_count"] - before_counts["promising_count"]),
    }


def _evaluate_ranked_usefulness(
    *,
    ranked_candidates: pd.DataFrame,
    full_frame: pd.DataFrame,
    market_meta: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
    family: str,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    merged = (
        ranked_candidates.loc[ranked_candidates.get("family", "").astype(str) == str(family)]
        .copy()
        .sort_values(["rank_score", "candidate_id"], ascending=[False, True])
    )
    rows: list[dict[str, Any]] = []
    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    min_trades = max(1, int(replay_gate.get("min_trade_count", 40)))
    for candidate in merged.head(max(1, int(candidate_limit))).to_dict(orient="records"):
        replay = run_candidate_replay(
            candidate=dict(candidate),
            config=config,
            symbol=symbol,
            frame=full_frame,
            market_meta=market_meta,
        )
        metrics = dict(replay.get("metrics", {}))
        trade_count = int(metrics.get("trade_count", 0))
        exp_lcb = float(metrics.get("exp_lcb", -1.0))
        aggregate_risk = float(candidate.get("aggregate_risk", 1.0))
        usefulness_class = _classify_usefulness_candidate(
            rank_score=float(candidate.get("rank_score", 0.0)),
            replay_trade_count=trade_count,
            replay_exp_lcb=exp_lcb,
            aggregate_risk=aggregate_risk,
            min_trade_count=min_trades,
        )
        first_death_stage = "survived" if usefulness_class != "junk" and trade_count >= max(1, min_trades // 2) else "replay"
        near_miss_distance = _replay_near_miss_distance(
            replay_trade_count=trade_count,
            replay_exp_lcb=exp_lcb,
            min_trade_count=min_trades,
        )
        rows.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "family": str(candidate.get("family", "")),
                "subfamily": str(candidate.get("subfamily", "")),
                "rank_score": float(candidate.get("rank_score", 0.0)),
                "aggregate_risk": aggregate_risk,
                "replay_trade_count": trade_count,
                "replay_exp_lcb": exp_lcb,
                "candidate_hierarchy": usefulness_class,
                "final_class": "promising_but_unproven" if usefulness_class == "promising_but_unproven" else "rejected",
                "first_death_stage": first_death_stage,
                "near_miss_distance": near_miss_distance,
            }
        )
    return rows


def _rank_candidates(
    *,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    ranking_frame: pd.DataFrame,
    ranking_profile: str,
) -> pd.DataFrame:
    ranked = score_candidates_with_ranker(
        candidates,
        labels,
        market_frame=ranking_frame.copy(),
        profile=str(ranking_profile),
    )
    return candidates.merge(ranked, on="candidate_id", how="inner")


def _classify_usefulness_candidate(
    *,
    rank_score: float,
    replay_trade_count: int,
    replay_exp_lcb: float,
    aggregate_risk: float,
    min_trade_count: int,
) -> str:
    if rank_score >= 0.42 and replay_trade_count >= max(8, min_trade_count // 2) and replay_exp_lcb >= -0.002 and aggregate_risk <= 0.62:
        return "promising_but_unproven"
    if rank_score >= 0.24 and replay_trade_count >= max(4, min_trade_count // 4) and replay_exp_lcb >= -0.012 and aggregate_risk <= 0.85:
        return "interesting_but_fragile"
    return "junk"


def _replay_near_miss_distance(
    *,
    replay_trade_count: int,
    replay_exp_lcb: float,
    min_trade_count: int,
) -> float:
    trade_gap = max(0.0, (float(min_trade_count) - float(max(0, replay_trade_count))) / float(max(1, min_trade_count)))
    exp_gap = max(0.0, 0.0 - float(replay_exp_lcb))
    return round(float(trade_gap + exp_gap), 6)


def _summarize_usefulness_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "candidate_count": 0,
            "useful_candidate_count": 0,
            "promising_count": 0,
            "mean_rank_score": 0.0,
            "mean_near_miss_distance": 0.0,
            "replay_death_fraction": 0.0,
            "near_miss_count": 0,
            "hierarchy_counts": {},
        }
    hierarchy_counts = Counter(str(row.get("candidate_hierarchy", "junk")) for row in rows)
    useful_count = int(sum(1 for row in rows if str(row.get("candidate_hierarchy", "")) in USEFUL_HIERARCHIES))
    promising_count = int(sum(1 for row in rows if str(row.get("final_class", "")) == "promising_but_unproven"))
    mean_rank = sum(float(row.get("rank_score", 0.0)) for row in rows) / len(rows)
    mean_near_miss = sum(float(row.get("near_miss_distance", 0.0)) for row in rows) / len(rows)
    replay_deaths = int(sum(1 for row in rows if str(row.get("first_death_stage", "")) == "replay"))
    near_miss_count = int(sum(1 for row in rows if 0.0 < float(row.get("near_miss_distance", 0.0)) <= 1.0))
    return {
        "candidate_count": int(len(rows)),
        "useful_candidate_count": useful_count,
        "promising_count": promising_count,
        "mean_rank_score": round(float(mean_rank), 6),
        "mean_near_miss_distance": round(float(mean_near_miss), 6),
        "replay_death_fraction": round(float(replay_deaths / max(1, len(rows))), 6),
        "near_miss_count": near_miss_count,
        "hierarchy_counts": dict(hierarchy_counts),
    }
