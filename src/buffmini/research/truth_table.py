"""Stage-100 multi-scope truth campaign utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any

from buffmini.research.campaign import evaluate_scope_campaign, select_campaign_families
from buffmini.research.transfer import discover_transfer_symbols
from buffmini.utils.hashing import stable_hash


TRUTH_MODES: dict[str, dict[str, Any]] = {
    "live_relaxed": {
        "requested_mode": "exploration",
        "auto_pin_resolved_end": False,
        "relax_continuity": True,
        "data_source_override": "live",
    },
    "live_strict": {
        "requested_mode": "evaluation",
        "auto_pin_resolved_end": True,
        "relax_continuity": False,
        "data_source_override": "live",
    },
    "canonical_strict": {
        "requested_mode": "evaluation",
        "auto_pin_resolved_end": True,
        "relax_continuity": False,
        "data_source_override": "canonical_eval",
    },
}

REGIME_BUCKETS: tuple[str, ...] = (
    "trend",
    "chop",
    "high_vol",
    "low_vol",
    "compression",
    "funding_extreme",
)


def run_multiscope_truth_campaign(
    config: dict[str, Any],
    *,
    candidate_limit_per_scope: int = 1,
) -> dict[str, Any]:
    symbols = discover_transfer_symbols(config)
    tier1_symbols = [symbol for symbol in symbols if symbol in {"BTC/USDT", "ETH/USDT"}]
    tier2_symbols = [symbol for symbol in symbols if symbol not in set(tier1_symbols)][:1]
    truth_symbols = tier1_symbols + tier2_symbols
    timeframes = ["30m", "1h", "4h"]
    families = select_campaign_families(limit=6)

    scope_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    by_scope_mode: dict[tuple[str, str, str], dict[str, Any]] = {}

    for symbol in truth_symbols:
        for timeframe in timeframes:
            for mode_name, mode_spec in TRUTH_MODES.items():
                summary = evaluate_scope_campaign(
                    config=config,
                    symbol=symbol,
                    timeframe=timeframe,
                    families=families,
                    candidate_limit=int(max(1, candidate_limit_per_scope)),
                    requested_mode=str(mode_spec["requested_mode"]),
                    auto_pin_resolved_end=bool(mode_spec["auto_pin_resolved_end"]),
                    relax_continuity=bool(mode_spec["relax_continuity"]),
                    evaluate_transfer=True,
                    ranking_profile="stage99_quality_acceleration",
                    data_source_override=str(mode_spec["data_source_override"]),
                )
                row = _build_scope_truth_row(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode_name,
                    summary=summary,
                )
                base_rows.append(row)
                by_scope_mode[(symbol, timeframe, mode_name)] = row

    for row in base_rows:
        relaxed_row = by_scope_mode.get((row["symbol"], row["timeframe"], "live_relaxed"), {})
        finalized = dict(row)
        finalized["truth_label"] = classify_scope_truth(row=finalized, relaxed_row=relaxed_row)
        scope_rows.append(finalized)
        regime_rows.extend(build_regime_truth_rows(finalized))

    truth_counts = Counter(str(row.get("truth_label", "")) for row in scope_rows)
    regime_truth_counts = Counter(str(row.get("truth_label", "")) for row in regime_rows)
    return {
        "families": families,
        "tier1_symbols": tier1_symbols,
        "tier2_symbols": tier2_symbols,
        "timeframes": timeframes,
        "modes": list(TRUTH_MODES),
        "candidate_limit_per_scope": int(max(1, candidate_limit_per_scope)),
        "scope_truth_rows": scope_rows,
        "regime_truth_rows": regime_rows,
        "truth_counts": dict(truth_counts),
        "regime_truth_counts": dict(regime_truth_counts),
        "summary_hash": stable_hash(
            {
                "scope_truth_rows": scope_rows,
                "truth_counts": dict(truth_counts),
                "regime_truth_counts": dict(regime_truth_counts),
            },
            length=16,
        ),
    }


def classify_scope_truth(
    *,
    row: dict[str, Any],
    relaxed_row: dict[str, Any] | None = None,
) -> str:
    relaxed = dict(relaxed_row or {})
    if bool(row.get("blocked", False)):
        return "data_blocks_interpretation"
    if int(row.get("robust_count", 0)) > 0:
        return "robust_signal_exists"
    if int(row.get("validated_count", 0)) > 0:
        return "validated_signal_exists"
    death_counts = dict(row.get("death_stage_counts", {}))
    if int(row.get("promising_count", 0)) > 0:
        if int(death_counts.get("replay", 0)) >= int(row.get("promising_count", 0)):
            return "replay_fragile_signal_only"
        return "weak_signal_exists"
    if str(row.get("mode", "")) != "live_relaxed" and int(relaxed.get("promising_count", 0)) > int(row.get("promising_count", 0)):
        if bool(relaxed.get("blocked", False)):
            return "data_blocks_interpretation"
        return "strict_evaluation_kills_things"
    dominant_failure_reason = str(row.get("dominant_failure_reason", ""))
    if dominant_failure_reason in {"transfer_fail", "regime_mismatch", "timing_instability"}:
        return "transfer_kills_things"
    if int(death_counts.get("transfer", 0)) > 0 and int(death_counts.get("transfer", 0)) >= int(max(1, sum(int(v) for v in death_counts.values())) // 2):
        return "transfer_kills_things"
    return "no_signal_exists"


def build_regime_truth_rows(scope_row: dict[str, Any]) -> list[dict[str, Any]]:
    evaluations = list(scope_row.get("evaluations", []))
    rows: list[dict[str, Any]] = []
    for regime in REGIME_BUCKETS:
        matched = [row for row in evaluations if regime in assign_regime_buckets(row)]
        blocked = bool(scope_row.get("blocked", False))
        rows.append(
            {
                "symbol": str(scope_row.get("symbol", "")),
                "timeframe": str(scope_row.get("timeframe", "")),
                "mode": str(scope_row.get("mode", "")),
                "regime": regime,
                "candidate_count": int(len(matched)),
                "promising_count": int(sum(1 for row in matched if str(row.get("final_class", "")) == "promising_but_unproven")),
                "validated_count": int(sum(1 for row in matched if str(row.get("candidate_hierarchy", "")) == "validated_candidate")),
                "robust_count": int(sum(1 for row in matched if str(row.get("final_class", "")) == "robust_candidate")),
                "blocked": blocked,
                "truth_label": str(scope_row.get("truth_label", "")) if blocked or matched else "no_targeted_candidate",
                "dominant_failure_reason": str(scope_row.get("dominant_failure_reason", "")),
            }
        )
    return rows


def assign_regime_buckets(candidate_row: dict[str, Any]) -> list[str]:
    expected_regime = str(candidate_row.get("expected_regime", "")).strip().lower()
    family = str(candidate_row.get("family", "")).strip()
    buckets: list[str] = []
    if expected_regime == "trend":
        buckets.append("trend")
    elif expected_regime == "range":
        buckets.append("chop")
    elif expected_regime == "compression":
        buckets.append("compression")
    elif expected_regime == "transition":
        buckets.append("high_vol")

    if family in {"structure_pullback_continuation", "multi_tf_disagreement_repair"}:
        buckets.extend(["trend", "low_vol"])
    elif family in {"liquidity_sweep_reversal", "exhaustion_mean_reversion"}:
        buckets.append("chop")
    elif family in {"squeeze_flow_breakout", "failed_breakout_reversal", "volatility_regime_transition"}:
        buckets.extend(["high_vol", "compression"])
    elif family == "funding_oi_imbalance_reversion":
        buckets.append("funding_extreme")

    ordered = []
    seen: set[str] = set()
    for bucket in buckets:
        if bucket in REGIME_BUCKETS and bucket not in seen:
            ordered.append(bucket)
            seen.add(bucket)
    return ordered


def _build_scope_truth_row(
    *,
    symbol: str,
    timeframe: str,
    mode: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    evaluations = list(summary.get("evaluations", []))
    death_stage_counts = Counter(str(row.get("first_death_stage", "")) for row in evaluations)
    dominant_failure_reason = ""
    reasons = dict(summary.get("dominant_failure_reasons", {}))
    if reasons:
        dominant_failure_reason = str(sorted(reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0])
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "blocked": bool(summary.get("blocked", False)),
        "blocked_reason": str(summary.get("blocked_reason", "")),
        "candidate_count": int(summary.get("candidate_count", 0)),
        "promising_count": int(summary.get("promising_count", 0)),
        "validated_count": int(summary.get("validated_count", 0)),
        "robust_count": int(summary.get("robust_count", 0)),
        "blocked_count": int(summary.get("blocked_count", 0)),
        "dominant_failure_reason": dominant_failure_reason,
        "death_stage_counts": dict(death_stage_counts),
        "evaluations": [
            {
                "candidate_id": str(row.get("candidate_id", "")),
                "family": str(row.get("family", "")),
                "expected_regime": str(row.get("expected_regime", "")),
                "final_class": str(row.get("final_class", "")),
                "candidate_hierarchy": str(row.get("candidate_hierarchy", "")),
                "first_death_stage": str(row.get("first_death_stage", "")),
                "death_reason": str(row.get("death_reason", "")),
                "replay_exp_lcb": float(row.get("replay_exp_lcb", -1.0)),
                "transfer_classification": str(row.get("transfer_classification", "")),
            }
            for row in evaluations
        ],
    }
