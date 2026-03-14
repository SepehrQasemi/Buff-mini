"""Stage-70 deterministic search space expansion with structured hypotheses."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


EXPANDED_FAMILIES: tuple[str, ...] = (
    "structure_pullback_continuation",
    "liquidity_sweep_reversal",
    "squeeze_flow_breakout",
    "crowded_side_squeeze",
    "flow_exhaustion_reversal",
    "regime_shift_entry",
    "volatility_reclaim_break",
    "trend_flush_recovery",
    "funding_imbalance_revert",
    "open_interest_divergence_break",
    "multi_tf_bias_alignment",
    "session_liquidity_rotation",
)

HYPOTHESIS_CONTEXTS: tuple[str, ...] = ("trend", "range", "transition", "vol_shock", "liquidity_rotation")
HYPOTHESIS_TRIGGERS: tuple[str, ...] = (
    "pullback_reclaim",
    "liquidity_sweep_reclaim",
    "compression_breakout",
    "funding_flush_reversion",
    "oi_divergence_break",
    "session_rotation_break",
)
HYPOTHESIS_CONFIRMATIONS: tuple[str, ...] = (
    "volume_expansion_confirm",
    "oi_expansion_confirm",
    "multi_tf_alignment_confirm",
    "momentum_slope_confirm",
    "none",
)
HYPOTHESIS_INVALIDATIONS: tuple[str, ...] = (
    "structure_break",
    "failed_reclaim",
    "flow_reversal",
    "volatility_recompression",
)
EXIT_FAMILIES: tuple[str, ...] = ("fixed_rr", "trailing_atr", "scale_out_then_trail", "time_exit")
TIME_STOP_BARS: tuple[int, ...] = (8, 12, 16, 24, 36, 48, 72)


def economic_fingerprint(record: dict[str, Any]) -> str:
    material = {
        "family": str(record.get("family", "")),
        "timeframe": str(record.get("timeframe", "")),
        "context": str(record.get("context", "")),
        "trigger": str(record.get("trigger", "")),
        "confirmation": str(record.get("confirmation", "")),
        "invalidation": str(record.get("invalidation", "")),
        "exit_family": str(record.get("exit_family", "")),
        "time_stop_bars": int(record.get("time_stop_bars", 0)),
    }
    return str(stable_hash(material, length=20))


def deduplicate_economic_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    if "economic_fingerprint" not in frame.columns:
        frame["economic_fingerprint"] = [economic_fingerprint(dict(row)) for row in frame.to_dict(orient="records")]
    frame["priority_seed"] = pd.to_numeric(frame.get("priority_seed", 0.0), errors="coerce").fillna(0.0)
    frame["novelty_score"] = pd.to_numeric(frame.get("novelty_score", 0.0), errors="coerce").fillna(0.0)
    ranked = frame.sort_values(["priority_seed", "novelty_score", "candidate_id"], ascending=[False, False, True]).drop_duplicates(
        subset=["economic_fingerprint"],
        keep="first",
    )
    return ranked.reset_index(drop=True)


def _build_hypothesis_row(*, idx: int, timeframes: list[str]) -> dict[str, Any]:
    family = EXPANDED_FAMILIES[idx % len(EXPANDED_FAMILIES)]
    timeframe = timeframes[idx % len(timeframes)]
    context = HYPOTHESIS_CONTEXTS[(idx // 2) % len(HYPOTHESIS_CONTEXTS)]
    trigger = HYPOTHESIS_TRIGGERS[(idx // 3) % len(HYPOTHESIS_TRIGGERS)]
    confirmation = HYPOTHESIS_CONFIRMATIONS[(idx // 5) % len(HYPOTHESIS_CONFIRMATIONS)]
    invalidation = HYPOTHESIS_INVALIDATIONS[(idx // 7) % len(HYPOTHESIS_INVALIDATIONS)]
    exit_family = EXIT_FAMILIES[(idx // 11) % len(EXIT_FAMILIES)]
    time_stop_bars = int(TIME_STOP_BARS[(idx // 13) % len(TIME_STOP_BARS)])
    hypothesis = {
        "context": context,
        "trigger": trigger,
        "participation": confirmation,
        "invalidation": invalidation,
        "exit_family": exit_family,
        "time_stop_bars": time_stop_bars,
    }
    fingerprint = economic_fingerprint(
        {
            "family": family,
            "timeframe": timeframe,
            "context": context,
            "trigger": trigger,
            "confirmation": confirmation,
            "invalidation": invalidation,
            "exit_family": exit_family,
            "time_stop_bars": time_stop_bars,
        }
    )
    priority_seed = float(round(0.25 + ((idx % 31) / 50.0), 8))
    novelty_score = float(round(0.20 + (((idx * 7) % 29) / 60.0), 8))
    candidate_id = f"s70_{stable_hash({'i': idx, 'fp': fingerprint}, length=16)}"
    return {
        "candidate_id": candidate_id,
        "family": family,
        "timeframe": timeframe,
        "context": context,
        "trigger": trigger,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "exit_family": exit_family,
        "time_stop_bars": int(time_stop_bars),
        "hypothesis": hypothesis,
        "economic_fingerprint": fingerprint,
        "priority_seed": priority_seed,
        "novelty_score": novelty_score,
    }


def generate_expanded_candidates(
    *,
    discovery_timeframes: list[str],
    budget_mode_selected: str,
    min_search_candidates: int = 2500,
    min_full_audit_candidates: int = 10000,
) -> pd.DataFrame:
    mode = str(budget_mode_selected).lower().strip()
    target = int(min_full_audit_candidates if mode == "full_audit" else min_search_candidates)
    timeframes = [str(v) for v in discovery_timeframes if str(v).strip()] or ["1h"]

    # Oversample then deduplicate by economic behavior.
    pool_size = int(max(target * 2, target + 128))
    pool_rows = [_build_hypothesis_row(idx=idx, timeframes=timeframes) for idx in range(pool_size)]
    deduped = deduplicate_economic_candidates(pd.DataFrame(pool_rows))

    # Guarantee target cardinality with deterministic suffix perturbations if needed.
    if len(deduped) < target:
        extra_rows: list[dict[str, Any]] = []
        idx = pool_size
        while len(deduped) + len(extra_rows) < target:
            row = _build_hypothesis_row(idx=idx, timeframes=timeframes)
            row["time_stop_bars"] = int(TIME_STOP_BARS[(idx + 3) % len(TIME_STOP_BARS)])
            row["economic_fingerprint"] = economic_fingerprint(row)
            row["candidate_id"] = f"s70_{stable_hash({'i': idx, 'fp': row['economic_fingerprint'], 'extra': 1}, length=16)}"
            extra_rows.append(row)
            idx += 1
        deduped = deduplicate_economic_candidates(pd.concat([deduped, pd.DataFrame(extra_rows)], ignore_index=True))

    out = deduped.sort_values(["priority_seed", "novelty_score", "candidate_id"], ascending=[False, False, True]).head(target).reset_index(drop=True)
    return out
