"""Reality isolation matrix and gate sensitivity diagnostics."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import pandas as pd

from buffmini.research.campaign import evaluate_scope_campaign
from buffmini.research.synthetic_lab import evaluate_detectability_suite


REALITY_ENVIRONMENTS = (
    "synthetic_clean_easy",
    "synthetic_clean_hard",
    "live_relaxed",
    "live_strict",
)


def evaluate_reality_matrix(config: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    results["synthetic_clean_easy"] = _summarize_synthetic_environment(
        evaluate_detectability_suite(config, seed=int((config.get("search", {}) or {}).get("seed", 42)), difficulty="easy")
    )
    results["synthetic_clean_hard"] = _summarize_synthetic_environment(
        evaluate_detectability_suite(config, seed=int((config.get("search", {}) or {}).get("seed", 42)), difficulty="hard")
    )

    live_relaxed_cfg = deepcopy(config)
    live_relaxed_cfg.setdefault("research_scope", {}).setdefault("primary_symbols", ["BTC/USDT"])
    families = list((live_relaxed_cfg.get("research_scope", {}) or {}).get("active_setup_families", []))
    results["live_relaxed"] = _summarize_live_environment(
        evaluate_scope_campaign(
            config=live_relaxed_cfg,
            symbol="BTC/USDT",
            timeframe="1h",
            families=families,
            candidate_limit=8,
            requested_mode="exploration",
            auto_pin_resolved_end=False,
            relax_continuity=True,
        )
    )
    results["live_strict"] = _summarize_live_environment(
        evaluate_scope_campaign(
            config=config,
            symbol="BTC/USDT",
            timeframe="1h",
            families=families,
            candidate_limit=8,
            requested_mode="evaluation",
            auto_pin_resolved_end=True,
            relax_continuity=False,
        )
    )

    sensitivity = build_gate_sensitivity(results=results, config=config)
    dominant_blockers = infer_dominant_blockers(results=results)
    return {
        "environments": results,
        "gate_sensitivity": sensitivity,
        "dominant_blockers": dominant_blockers,
    }


def build_gate_sensitivity(*, results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    walkforward_gate = dict(config.get("promotion_gates", {}).get("walkforward", {}))
    return {
        "replay": {
            "baseline_min_trade_count": int(replay_gate.get("min_trade_count", 40)),
            "baseline_min_exp_lcb": float(replay_gate.get("min_exp_lcb", 0.0)),
            "survivor_counts": {
                env: _count_replay_survivors(payload, min_trade_count=int(replay_gate.get("min_trade_count", 40)), min_exp_lcb=float(replay_gate.get("min_exp_lcb", 0.0)))
                for env, payload in results.items()
            },
            "looser_trade_count": {
                env: _count_replay_survivors(payload, min_trade_count=max(1, int(replay_gate.get("min_trade_count", 40)) // 2), min_exp_lcb=float(replay_gate.get("min_exp_lcb", 0.0)) - 0.002)
                for env, payload in results.items()
            },
            "stricter_trade_count": {
                env: _count_replay_survivors(payload, min_trade_count=int(replay_gate.get("min_trade_count", 40)) + 10, min_exp_lcb=float(replay_gate.get("min_exp_lcb", 0.0)) + 0.001)
                for env, payload in results.items()
            },
        },
        "walkforward": {
            "baseline_min_usable_windows": int(walkforward_gate.get("min_usable_windows", 3)),
            "survivor_counts": {
                env: _count_walkforward_survivors(payload, min_windows=int(walkforward_gate.get("min_usable_windows", 3)))
                for env, payload in results.items()
            },
            "looser": {
                env: _count_walkforward_survivors(payload, min_windows=max(1, int(walkforward_gate.get("min_usable_windows", 3)) - 1))
                for env, payload in results.items()
            },
            "stricter": {
                env: _count_walkforward_survivors(payload, min_windows=int(walkforward_gate.get("min_usable_windows", 3)) + 1)
                for env, payload in results.items()
            },
        },
        "monte_carlo": {
            "survivor_counts": {env: _count_stage_survivors(payload, stage="monte_carlo") for env, payload in results.items()},
        },
        "transfer": {
            "survivor_counts": {env: _count_transfer_survivors(payload) for env, payload in results.items()},
        },
        "continuity": {
            "blocked_counts": {
                env: int(payload.get("blocked_count", 0)) if env.startswith("live_") else 0
                for env, payload in results.items()
            }
        },
    }


def infer_dominant_blockers(*, results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for env, payload in results.items():
        reasons = payload.get("dominant_failure_reasons", {}) or {}
        if isinstance(reasons, dict):
            for reason, count in reasons.items():
                key = _map_reason_to_blocker(str(reason), env=env)
                counter[key] += int(count)
    return [{"blocker": key, "count": int(value)} for key, value in counter.most_common()]


def _summarize_synthetic_environment(summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = list(summary.get("evaluations", []))
    robust_count = int(sum(1 for row in evaluations if str(row.get("candidate_class", "")) == "validated_candidate"))
    dominant_failure_reasons = Counter()
    for row in evaluations:
        if str(row.get("candidate_class", "")) == "rejected":
            if float(row.get("replay_exp_lcb", -1.0)) < 0.0:
                dominant_failure_reasons["replay_exp_lcb"] += 1
            else:
                dominant_failure_reasons["ranking_filter"] += 1
    return {
        "candidate_count": int(summary.get("candidate_count", 0)),
        "promising_count": int(summary.get("candidate_classes", {}).get("promising_but_unproven", 0)),
        "validated_count": int(summary.get("candidate_classes", {}).get("validated_candidate", 0)),
        "robust_count": robust_count,
        "blocked_count": 0,
        "dominant_failure_reasons": dict(dominant_failure_reasons),
        "first_death_stage_counts": {
            "ranking_filter": int(sum(1 for row in evaluations if str(row.get("candidate_class", "")) == "rejected")),
        },
        "near_miss_inventory": [
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "near_miss_distance": float(max(0.0, -float(row.get("replay_exp_lcb", 0.0)))),
                "death_stage": "ranking_filter" if str(row.get("candidate_class", "")) == "rejected" else "survived",
                "death_reason": "replay_exp_lcb" if float(row.get("replay_exp_lcb", 0.0)) < 0.0 else ""
            }
            for row in evaluations
        ],
        "evaluations": evaluations,
        "summary_hash": str(summary.get("summary_hash", "")),
    }


def _summarize_live_environment(summary: dict[str, Any]) -> dict[str, Any]:
    evaluations = list(summary.get("evaluations", []))
    death_counts = Counter(str(row.get("first_death_stage", "unknown")) for row in evaluations)
    near_miss = sorted(
        [
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "near_miss_distance": float(row.get("near_miss_distance", 0.0)),
                "death_stage": str(row.get("first_death_stage", "")),
                "death_reason": str(row.get("death_reason", "")),
            }
            for row in evaluations
            if str(row.get("first_death_stage", "")) != "survived"
        ],
        key=lambda row: (float(row.get("near_miss_distance", 9.0)), str(row.get("candidate_id", ""))),
    )
    return {
        "candidate_count": int(summary.get("candidate_count", 0)),
        "promising_count": int(summary.get("promising_count", 0)),
        "validated_count": int(summary.get("validated_count", 0)),
        "robust_count": int(summary.get("robust_count", 0)),
        "blocked_count": int(summary.get("blocked_count", 0)),
        "dominant_failure_reasons": dict(summary.get("dominant_failure_reasons", {})),
        "first_death_stage_counts": dict(death_counts),
        "near_miss_inventory": near_miss[:10],
        "evaluations": evaluations,
        "blocked_reason": str(summary.get("blocked_reason", "")),
        "mode_summary": dict(summary.get("mode_summary", {})),
    }


def _count_replay_survivors(payload: dict[str, Any], *, min_trade_count: int, min_exp_lcb: float) -> int:
    rows = list(payload.get("evaluations", []))
    return int(sum(1 for row in rows if int(row.get("replay_trade_count", 0)) >= int(min_trade_count) and float(row.get("replay_exp_lcb", -1.0)) >= float(min_exp_lcb)))


def _count_walkforward_survivors(payload: dict[str, Any], *, min_windows: int) -> int:
    rows = list(payload.get("evaluations", []))
    return int(sum(1 for row in rows if int(row.get("walkforward_usable_windows", 0)) >= int(min_windows)))


def _count_stage_survivors(payload: dict[str, Any], *, stage: str) -> int:
    rows = list(payload.get("evaluations", []))
    if stage == "monte_carlo":
        return int(sum(1 for row in rows if str(row.get("first_death_stage", "")) not in {"replay", "walkforward", "monte_carlo"}))
    return 0


def _count_transfer_survivors(payload: dict[str, Any]) -> int:
    rows = list(payload.get("evaluations", []))
    return int(sum(1 for row in rows if str(row.get("transfer_classification", "")) in {"transferable", "partially_transferable"}))


def _map_reason_to_blocker(reason: str, *, env: str) -> str:
    text = reason.lower()
    if env == "live_strict" and ("gap" in text or "continuity" in text):
        return "data_canonicalization"
    if "transfer" in text:
        return "transfer_limitation"
    if text in {"ranking_filter", "exp_lcb", "replay_exp_lcb"}:
        return "ranking_funnel_pressure"
    if "walkforward" in text:
        return "evaluation_strictness"
    if "stage" in text or "replay" in text:
        return "generator_depth"
    return "scope_limitation"
