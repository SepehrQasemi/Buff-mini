"""Stage-97 relaxed-to-strict bridge for promising candidates."""

from __future__ import annotations

from collections import Counter
from typing import Any

from buffmini.research.campaign import evaluate_candidate_batch, evaluate_scope_campaign
from buffmini.research.modes import build_mode_context
from buffmini.research.transfer import discover_transfer_symbols
from buffmini.stage51.scope import resolve_research_scope
from buffmini.utils.hashing import stable_hash
from buffmini.validation import load_candidate_market_frame


def build_relaxed_to_strict_bridge(
    config: dict[str, Any],
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    candidate_limit_per_scope: int = 1,
) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    use_symbols = list(symbols or (scope.get("primary_symbols") or ["BTC/USDT"]))[:1]
    use_timeframes = list(timeframes or ["30m", "1h", "4h"])
    families = list(scope.get("active_setup_families") or [])

    relaxed_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []

    for symbol in use_symbols:
        for timeframe in use_timeframes:
            relaxed = evaluate_scope_campaign(
                config=config,
                symbol=symbol,
                timeframe=timeframe,
                families=families,
                candidate_limit=int(candidate_limit_per_scope),
                requested_mode="exploration",
                auto_pin_resolved_end=False,
                relax_continuity=True,
                evaluate_transfer=False,
                ranking_profile="stage99_quality_acceleration",
            )
            ranked_frame = relaxed.get("ranked_frame")
            if relaxed.get("blocked", False):
                continue
            chosen = _select_bridge_candidates(relaxed)
            if not chosen:
                continue
            relaxed_rows.extend(
                [
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        **row,
                    }
                    for row in chosen
                ]
            )
            candidate_lookup = {
                str(row.get("candidate_id", "")): dict(row)
                for row in getattr(ranked_frame, "to_dict", lambda **_: [])(orient="records")
            }
            strict_cfg, _ = build_mode_context(config, requested_mode="evaluation", auto_pin_resolved_end=True)
            strict_frame, strict_meta = load_candidate_market_frame(strict_cfg, symbol=symbol, timeframe=timeframe)
            if strict_frame.empty or bool(strict_meta.get("runtime_truth_blocked", False)) or bool(strict_meta.get("continuity_blocked", False)):
                for row in chosen:
                    bridge_rows.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "candidate_id": str(row.get("candidate_id", "")),
                            "family": str(row.get("family", "")),
                            "relaxed_final_class": str(row.get("final_class", "")),
                            "strict_final_class": "blocked",
                            "bridge_classification": "killed_by_data",
                            "strict_kill_reason": str(strict_meta.get("continuity_reason", "") or strict_meta.get("runtime_truth_reason", "")),
                        }
                    )
                continue

            transfer_symbols = discover_transfer_symbols(strict_cfg, primary_symbol=symbol)
            transfer_symbol = next((item for item in transfer_symbols if item != symbol), "")
            strict_candidates = [candidate_lookup[str(row.get("candidate_id", ""))] for row in chosen if str(row.get("candidate_id", "")) in candidate_lookup]
            strict_rows = evaluate_candidate_batch(
                candidates=strict_candidates,
                config=strict_cfg,
                symbol=symbol,
                frame=strict_frame,
                market_meta=strict_meta,
                transfer_symbol=transfer_symbol,
                evaluate_transfer=True,
            )
            strict_lookup = {str(row.get("candidate_id", "")): row for row in strict_rows}
            for relaxed_row in chosen:
                strict_row = strict_lookup.get(str(relaxed_row.get("candidate_id", "")), {})
                bridge_rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "candidate_id": str(relaxed_row.get("candidate_id", "")),
                        "family": str(relaxed_row.get("family", "")),
                        "relaxed_final_class": str(relaxed_row.get("final_class", "")),
                        "strict_final_class": str(strict_row.get("final_class", "missing")),
                        "relaxed_first_death_stage": str(relaxed_row.get("first_death_stage", "")),
                        "strict_first_death_stage": str(strict_row.get("first_death_stage", "")),
                        "relaxed_replay_exp_lcb": float(relaxed_row.get("replay_exp_lcb", -1.0)),
                        "strict_replay_exp_lcb": float(strict_row.get("replay_exp_lcb", -1.0)),
                        "bridge_classification": classify_bridge_transition(relaxed_row=relaxed_row, strict_row=strict_row),
                        "strict_kill_reason": str(strict_row.get("death_reason", "")),
                    }
                )

    classification_counts = Counter(str(row.get("bridge_classification", "")) for row in bridge_rows)
    return {
        "symbols": use_symbols,
        "timeframes": use_timeframes,
        "candidate_limit_per_scope": int(candidate_limit_per_scope),
        "relaxed_candidate_count": int(len(relaxed_rows)),
        "bridge_rows": bridge_rows,
        "classification_counts": dict(classification_counts),
        "summary_hash": stable_hash(
            {
                "symbols": use_symbols,
                "timeframes": use_timeframes,
                "classification_counts": dict(classification_counts),
            },
            length=16,
        ),
    }


def classify_bridge_transition(*, relaxed_row: dict[str, Any], strict_row: dict[str, Any]) -> str:
    if not strict_row:
        return "killed_by_data"
    strict_class = str(strict_row.get("final_class", ""))
    strict_hierarchy = str(strict_row.get("candidate_hierarchy", ""))
    strict_stage = str(strict_row.get("first_death_stage", ""))
    if strict_class == "promising_but_unproven":
        return "survive" if strict_stage not in {"", "replay"} else "become_weaker"
    if strict_hierarchy == "interesting_but_fragile":
        return "become_weaker"
    if strict_stage == "transfer":
        return "killed_by_transfer"
    if strict_stage == "walkforward":
        return "killed_by_walkforward"
    if strict_stage == "replay":
        return "killed_by_replay"
    return "become_weaker"


def _select_bridge_candidates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    evaluations = list(summary.get("evaluations", []))
    promising = [row for row in evaluations if str(row.get("final_class", "")) == "promising_but_unproven"]
    if promising:
        return promising
    fallback = [
        row
        for row in evaluations
        if str(row.get("candidate_hierarchy", "")) in {"interesting_but_fragile", "promising_but_unproven"}
    ]
    return fallback[: max(1, int(summary.get("promising_count", 0)) or 1)]
