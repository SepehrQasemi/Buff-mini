"""Stage-102 bounded robustness rescue attempts for top promising candidates."""

from __future__ import annotations

from typing import Any

from buffmini.research.campaign import evaluate_candidate_record, evaluate_scope_campaign, select_campaign_families
from buffmini.research.modes import build_mode_context
from buffmini.research.transfer import discover_transfer_symbols
from buffmini.utils.hashing import stable_hash
from buffmini.validation import load_candidate_market_frame


RESCUE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "shorter_hold_horizon",
        "candidate_overrides": {"geometry": {"expected_hold_bars": 6}},
    },
    {
        "name": "tighter_invalidation",
        "candidate_overrides": {"geometry": {"stop_distance_pct": 0.0048, "first_target_pct": 0.0090}},
    },
    {
        "name": "extension_exit",
        "candidate_overrides": {"target_logic": "extension"},
    },
    {
        "name": "reversion_exit",
        "candidate_overrides": {"target_logic": "reversion"},
    },
    {
        "name": "stricter_participation",
        "replay_options": {"signal_keep_ratio": 0.7},
    },
    {
        "name": "diagnostic_transfer_relaxed",
        "evaluate_transfer": False,
        "diagnostic_only": True,
    },
)


def run_rescue_attempts(
    config: dict[str, Any],
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    candidate_limit: int = 3,
) -> dict[str, Any]:
    families = select_campaign_families(limit=6)
    source_scope = evaluate_scope_campaign(
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=int(max(1, candidate_limit)),
        requested_mode="evaluation",
        auto_pin_resolved_end=True,
        relax_continuity=False,
        evaluate_transfer=True,
        ranking_profile="stage99_quality_acceleration",
        data_source_override="canonical_eval",
    )
    if bool(source_scope.get("blocked", False)):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candidate_limit_reviewed": int(max(1, candidate_limit)),
            "candidate_rows": [],
            "classification_counts": {"data_blocked": int(max(1, candidate_limit))},
            "blocked": True,
            "blocked_reason": str(source_scope.get("blocked_reason", "")),
            "summary_hash": stable_hash({"blocked_reason": source_scope.get("blocked_reason", "")}, length=16),
        }

    ranked_lookup = {
        str(row.get("candidate_id", "")): dict(row)
        for row in getattr(source_scope.get("ranked_frame"), "to_dict", lambda **_: [])(orient="records")
    }
    chosen = _select_candidates(source_scope=source_scope, limit=int(max(1, candidate_limit)))
    effective_cfg, _ = build_mode_context(config, requested_mode="evaluation", auto_pin_resolved_end=True)
    frame, market_meta = load_candidate_market_frame(effective_cfg, symbol=symbol, timeframe=timeframe)
    transfer_symbols = discover_transfer_symbols(effective_cfg, primary_symbol=symbol)
    transfer_symbol = next((value for value in transfer_symbols if value != symbol), "")

    candidate_rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for chosen_row in chosen:
        candidate = dict(ranked_lookup.get(str(chosen_row.get("candidate_id", "")), {}))
        if not candidate:
            continue
        base = evaluate_candidate_record(
            candidate=candidate,
            config=effective_cfg,
            symbol=symbol,
            frame=frame,
            market_meta=market_meta,
            transfer_symbol=transfer_symbol,
            evaluate_transfer=True,
        )
        variants = []
        for spec in RESCUE_VARIANTS:
            variant_result = evaluate_candidate_record(
                candidate=candidate,
                config=effective_cfg,
                symbol=symbol,
                frame=frame,
                market_meta=market_meta,
                transfer_symbol=transfer_symbol,
                evaluate_transfer=bool(spec.get("evaluate_transfer", True)),
                candidate_overrides=dict(spec.get("candidate_overrides", {})),
                replay_options=dict(spec.get("replay_options", {})),
            )
            variants.append(
                {
                    "variant": str(spec["name"]),
                    "diagnostic_only": bool(spec.get("diagnostic_only", False)),
                    "final_class": str(variant_result.get("final_class", "")),
                    "candidate_hierarchy": str(variant_result.get("candidate_hierarchy", "")),
                    "first_death_stage": str(variant_result.get("first_death_stage", "")),
                    "death_reason": str(variant_result.get("death_reason", "")),
                    "replay_exp_lcb": float(variant_result.get("replay_exp_lcb", -1.0)),
                    "robustness_level": int(variant_result.get("robustness_level", 0)),
                    "transfer_classification": str(variant_result.get("transfer_classification", "")),
                }
            )
        classification = classify_rescue_outcome(base=base, variants=variants)
        class_counts[classification] = class_counts.get(classification, 0) + 1
        candidate_rows.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "family": str(candidate.get("family", "")),
                "expected_regime": str(candidate.get("expected_regime", "")),
                "base_final_class": str(base.get("final_class", "")),
                "base_first_death_stage": str(base.get("first_death_stage", "")),
                "base_replay_exp_lcb": float(base.get("replay_exp_lcb", -1.0)),
                "classification": classification,
                "variants": variants,
            }
        )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candidate_limit_reviewed": int(max(1, candidate_limit)),
        "candidate_rows": candidate_rows,
        "classification_counts": class_counts,
        "blocked": False,
        "blocked_reason": "",
        "summary_hash": stable_hash(
            {
                "candidate_rows": candidate_rows,
                "classification_counts": class_counts,
            },
            length=16,
        ),
    }


def classify_rescue_outcome(*, base: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    if not variants:
        return "not_rescueable"
    if str(base.get("validation_state", "")).startswith("CONTINUITY") or str(base.get("validation_state", "")).startswith("RUNTIME_TRUTH"):
        return "data_blocked"
    non_diagnostic = [row for row in variants if not bool(row.get("diagnostic_only", False))]
    best_non_diagnostic = max(
        non_diagnostic,
        key=lambda row: (int(row.get("robustness_level", 0)), float(row.get("replay_exp_lcb", -1.0))),
        default={},
    )
    if str(best_non_diagnostic.get("final_class", "")) in {"promising_but_unproven", "robust_candidate"} and str(best_non_diagnostic.get("first_death_stage", "")) not in {"replay", ""}:
        return "rescueable"
    if non_diagnostic and all(str(row.get("first_death_stage", "")) == "replay" for row in non_diagnostic) and float(best_non_diagnostic.get("replay_exp_lcb", -1.0)) < 0.0:
        return "still_generator_weak"
    transfer_relaxed = next((row for row in variants if str(row.get("variant", "")) == "diagnostic_transfer_relaxed"), {})
    if (
        str(transfer_relaxed.get("final_class", "")) == "promising_but_unproven"
        and str(base.get("transfer_classification", "")) not in {"transferable", "partially_transferable"}
        and str(base.get("first_death_stage", "")) != "replay"
    ):
        return "transfer_blocked"
    return "not_rescueable"


def _select_candidates(*, source_scope: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    evaluations = list(source_scope.get("evaluations", []))
    promising = [row for row in evaluations if str(row.get("final_class", "")) == "promising_but_unproven"]
    chosen = promising or evaluations
    return chosen[: max(1, int(limit))]
