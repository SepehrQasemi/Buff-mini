"""Controlled scope expansion ladder diagnostics."""

from __future__ import annotations

from typing import Any

from buffmini.research.campaign import evaluate_scope_campaign, select_campaign_families
from buffmini.research.transfer import discover_transfer_symbols
from buffmini.stage51.scope import resolve_research_scope
from buffmini.utils.hashing import stable_hash


TIMEFRAME_LADDER: tuple[str, ...] = ("15m", "30m", "1h", "4h")
REGIME_LADDER: tuple[str, ...] = (
    "trend",
    "chop",
    "high_vol",
    "low_vol",
    "funding_extreme",
    "compression",
)


def evaluate_scope_ladder(
    config: dict[str, Any],
    *,
    feedback: dict[str, Any] | None = None,
    candidate_limit: int = 3,
) -> dict[str, Any]:
    scope = resolve_research_scope(config)
    families = select_campaign_families(feedback, limit=6)
    symbols = discover_transfer_symbols(config)
    tier1 = [symbol for symbol in symbols if symbol in {"BTC/USDT", "ETH/USDT"}]
    tier2 = [symbol for symbol in symbols if symbol not in set(tier1)][:2]
    rows: list[dict[str, Any]] = []
    for asset_tier, tier_symbols in (("tier1", tier1), ("tier2", tier2)):
        for symbol in tier_symbols:
            for timeframe in TIMEFRAME_LADDER:
                summary = evaluate_scope_campaign(
                    config=config,
                    symbol=symbol,
                    timeframe=timeframe,
                    families=families,
                    candidate_limit=int(max(1, candidate_limit)),
                    requested_mode="exploration",
                    auto_pin_resolved_end=False,
                    relax_continuity=True,
                    evaluate_transfer=False,
                )
                rows.append(
                    {
                        "asset_tier": asset_tier,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "candidate_density": int(summary.get("candidate_count", 0)),
                        "promising_density": int(summary.get("promising_count", 0)),
                        "validated_density": int(summary.get("validated_count", 0)),
                        "robust_density": int(summary.get("robust_count", 0)),
                        "blocked_density": int(summary.get("blocked_count", 0)),
                        "dominant_failure_reason": _dominant_failure_reason(summary),
                    }
                )
    regime_map = _regime_scope_map(rows=rows, scope=scope)
    return {
        "rows": rows,
        "tier1_symbols": tier1,
        "tier2_symbols": tier2,
        "candidate_limit_per_scope": int(max(1, candidate_limit)),
        "regime_ladder": list(REGIME_LADDER),
        "regime_scope_map": regime_map,
        "summary_hash": stable_hash(
            {
                "row_count": len(rows),
                "tier1_symbols": tier1,
                "tier2_symbols": tier2,
                "dominant_failure_reasons": [row["dominant_failure_reason"] for row in rows],
            },
            length=16,
        ),
    }


def _dominant_failure_reason(summary: dict[str, Any]) -> str:
    reasons = dict(summary.get("dominant_failure_reasons", {}))
    if not reasons:
        return ""
    return str(sorted(reasons.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0])


def _regime_scope_map(*, rows: list[dict[str, Any]], scope: dict[str, Any]) -> list[dict[str, Any]]:
    active_families = [str(item) for item in scope.get("active_setup_families", [])]
    out: list[dict[str, Any]] = []
    for regime in REGIME_LADDER:
        out.append(
            {
                "regime": regime,
                "family_count": int(sum(1 for family in active_families if _family_matches_regime(family, regime))),
                "timeframe_count": int(sum(1 for row in rows if _timeframe_matches_regime(str(row.get("timeframe", "")), regime))),
            }
        )
    return out


def _family_matches_regime(family: str, regime: str) -> bool:
    family = str(family)
    regime = str(regime)
    if regime == "trend":
        return family in {"structure_pullback_continuation", "multi_tf_disagreement_repair"}
    if regime == "chop":
        return family in {"liquidity_sweep_reversal", "exhaustion_mean_reversion"}
    if regime == "high_vol":
        return family in {"squeeze_flow_breakout", "failed_breakout_reversal"}
    if regime == "low_vol":
        return family in {"structure_pullback_continuation", "multi_tf_disagreement_repair"}
    if regime == "funding_extreme":
        return family in {"funding_oi_imbalance_reversion"}
    if regime == "compression":
        return family in {"squeeze_flow_breakout", "volatility_regime_transition"}
    return False


def _timeframe_matches_regime(timeframe: str, regime: str) -> bool:
    if regime in {"trend", "low_vol"}:
        return timeframe in {"1h", "4h"}
    if regime in {"high_vol", "compression"}:
        return timeframe in {"15m", "30m", "1h"}
    return True
