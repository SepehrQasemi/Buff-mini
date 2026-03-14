"""Mechanism-based hypothesis generation for discovery and campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class MechanismSpec:
    family: str
    contexts: tuple[str, ...]
    triggers: tuple[str, ...]
    confirmations: tuple[str, ...]
    participation_styles: tuple[str, ...]
    invalidations: tuple[str, ...]
    risk_models: tuple[str, ...]
    exit_families: tuple[str, ...]
    time_stops: tuple[int, ...]
    session_filters: tuple[str, ...]
    modules: tuple[str, ...]
    transfer_risk_prior: float


MECHANISM_SPECS: tuple[MechanismSpec, ...] = (
    MechanismSpec(
        family="structure_pullback_continuation",
        contexts=("trend", "transition"),
        triggers=("pullback_reclaim", "micro_break_reclaim"),
        confirmations=("volume_expansion_confirm", "flow_continuation_confirm"),
        participation_styles=("aligned_flow", "break_reclaim", "impulse_rejoin"),
        invalidations=("structure_break", "failed_reclaim"),
        risk_models=("trend_pullback",),
        exit_families=("scale_out_then_trail", "trailing_atr"),
        time_stops=(8, 12, 16),
        session_filters=("any_session", "ldn_ny_overlap"),
        modules=("structure_engine", "flow_regime_engine", "mtf_bias_completion"),
        transfer_risk_prior=0.22,
    ),
    MechanismSpec(
        family="liquidity_sweep_reversal",
        contexts=("range", "transition", "liquidity_rotation"),
        triggers=("liquidity_sweep_reclaim", "pool_violation_reclaim"),
        confirmations=("rejection_close_confirm", "flow_exhaustion_confirm"),
        participation_styles=("wick_rejection", "close_back_inside", "delayed_reclaim"),
        invalidations=("failed_reclaim", "flow_reversal"),
        risk_models=("sweep_reversal",),
        exit_families=("fixed_rr", "breakeven_then_fixed"),
        time_stops=(4, 6, 8),
        session_filters=("any_session", "session_extremes"),
        modules=("liquidity_map", "flow_regime_engine", "trade_geometry_layer"),
        transfer_risk_prior=0.28,
    ),
    MechanismSpec(
        family="squeeze_flow_breakout",
        contexts=("vol_shock", "transition"),
        triggers=("compression_breakout", "range_expansion_break"),
        confirmations=("volume_expansion_confirm", "flow_burst_confirm"),
        participation_styles=("breakout_chase", "retest_hold", "opening_drive"),
        invalidations=("volatility_recompression", "failed_breakout"),
        risk_models=("breakout_follow_through",),
        exit_families=("trailing_atr", "scale_out_then_trail"),
        time_stops=(10, 16, 24),
        session_filters=("any_session", "high_liquidity_hours"),
        modules=("volatility_regime_engine", "flow_regime_engine", "mtf_bias_completion"),
        transfer_risk_prior=0.24,
    ),
    MechanismSpec(
        family="failed_breakout_reversal",
        contexts=("transition", "vol_shock"),
        triggers=("failed_breakout_snapback", "fake_breakout_reversal"),
        confirmations=("exhaustion_confirm", "reclaim_confirm"),
        participation_styles=("snapback_entry", "close_back_inside", "reclaim_retest"),
        invalidations=("breakout_reacceptance", "trend_continuation"),
        risk_models=("failed_breakout_reversal",),
        exit_families=("fixed_rr", "breakeven_then_fixed"),
        time_stops=(4, 6, 10),
        session_filters=("any_session", "session_extremes"),
        modules=("liquidity_map", "volatility_regime_engine", "flow_regime_engine"),
        transfer_risk_prior=0.35,
    ),
    MechanismSpec(
        family="exhaustion_mean_reversion",
        contexts=("range", "vol_shock"),
        triggers=("stretch_extreme_revert", "exhaustion_reversal"),
        confirmations=("flow_exhaustion_confirm", "volatility_fade_confirm"),
        participation_styles=("fade_extreme", "close_reclaim", "mean_snap"),
        invalidations=("trend_resume", "new_extreme_break"),
        risk_models=("mean_reversion",),
        exit_families=("fixed_rr", "time_exit"),
        time_stops=(4, 8, 12),
        session_filters=("any_session", "late_session"),
        modules=("flow_regime_engine", "trade_geometry_layer"),
        transfer_risk_prior=0.40,
    ),
    MechanismSpec(
        family="funding_oi_imbalance_reversion",
        contexts=("crowding", "transition"),
        triggers=("funding_flush_reversion", "oi_divergence_revert"),
        confirmations=("crowding_extreme_confirm", "sentiment_reversal_confirm"),
        participation_styles=("sentiment_snap", "imbalance_fade", "delayed_reversal"),
        invalidations=("crowding_acceleration", "trend_resume"),
        risk_models=("crowding_reversion",),
        exit_families=("fixed_rr", "time_exit"),
        time_stops=(6, 12, 18),
        session_filters=("any_session", "high_liquidity_hours"),
        modules=("crowding_layer", "flow_regime_engine"),
        transfer_risk_prior=0.48,
    ),
    MechanismSpec(
        family="volatility_regime_transition",
        contexts=("transition", "vol_shock"),
        triggers=("compression_to_expansion", "vol_regime_reclaim"),
        confirmations=("breakout_readiness_confirm", "momentum_slope_confirm"),
        participation_styles=("anticipatory_break", "confirmation_break", "retest_break"),
        invalidations=("recompression", "failed_expansion"),
        risk_models=("vol_transition",),
        exit_families=("trailing_atr", "time_exit"),
        time_stops=(8, 14, 20),
        session_filters=("any_session", "high_liquidity_hours"),
        modules=("volatility_regime_engine", "structure_engine"),
        transfer_risk_prior=0.30,
    ),
    MechanismSpec(
        family="multi_tf_disagreement_repair",
        contexts=("transition", "trend"),
        triggers=("htf_ltf_disagreement_repair", "bias_realignment_break"),
        confirmations=("structure_realign_confirm", "flow_continuation_confirm"),
        participation_styles=("alignment_reclaim", "late_confirmation", "pullback_after_realign"),
        invalidations=("misalignment_persists", "htf_bias_flip"),
        risk_models=("multi_tf_repair",),
        exit_families=("trailing_atr", "scale_out_then_trail"),
        time_stops=(8, 12, 18),
        session_filters=("any_session", "ldn_ny_overlap"),
        modules=("mtf_bias_completion", "structure_engine", "flow_regime_engine"),
        transfer_risk_prior=0.26,
    ),
)


def mechanism_families() -> tuple[str, ...]:
    return tuple(spec.family for spec in MECHANISM_SPECS)


def generate_mechanism_source_candidates(
    *,
    discovery_timeframes: list[str],
    budget_mode_selected: str,
    active_families: list[str] | None = None,
    target_min_candidates: int | None = None,
) -> pd.DataFrame:
    """Generate economically structured source candidates with bounded diversity."""

    families_filter = {str(v).strip() for v in (active_families or []) if str(v).strip()}
    specs = [spec for spec in MECHANISM_SPECS if not families_filter or spec.family in families_filter]
    timeframes = [str(v).strip() for v in discovery_timeframes if str(v).strip()] or ["1h"]
    mode = str(budget_mode_selected).strip().lower()
    target = int(target_min_candidates or (10000 if mode == "full_audit" else 2500))

    rows: list[dict[str, object]] = []
    idx = 0
    for spec in specs:
        for timeframe, context, trigger, confirmation, participation_style, invalidation, risk_model, exit_family, time_stop_bars, session_filter in product(
            timeframes,
            spec.contexts,
            spec.triggers,
            spec.confirmations,
            spec.participation_styles,
            spec.invalidations,
            spec.risk_models,
            spec.exit_families,
            spec.time_stops,
            spec.session_filters,
        ):
            mechanism_signature = stable_hash(
                {
                    "family": spec.family,
                    "timeframe": timeframe,
                    "context": context,
                    "trigger": trigger,
                    "confirmation": confirmation,
                    "participation_style": participation_style,
                    "invalidation": invalidation,
                    "risk_model": risk_model,
                    "exit_family": exit_family,
                    "time_stop_bars": int(time_stop_bars),
                    "session_filter": session_filter,
                },
                length=20,
            )
            rows.append(
                {
                    "candidate_id": f"s70_{stable_hash({'i': idx, 'fp': mechanism_signature}, length=16)}",
                    "family": spec.family,
                    "mechanism_family": spec.family,
                    "timeframe": timeframe,
                    "context": context,
                    "trigger": trigger,
                    "confirmation": confirmation,
                    "participation": participation_style,
                    "participation_style": participation_style,
                    "invalidation": invalidation,
                    "risk_model": risk_model,
                    "exit_family": exit_family,
                    "time_stop_bars": int(time_stop_bars),
                    "session_filter": session_filter,
                    "modules": list(spec.modules),
                    "transfer_risk_prior": float(spec.transfer_risk_prior),
                    "mechanism_signature": mechanism_signature,
                    "priority_seed": float(round(0.28 + ((idx % 23) / 40.0), 6)),
                    "novelty_score": float(round(0.22 + (((idx * 7) % 31) / 55.0), 6)),
                    "beam_score": float(round(0.35 + (((idx * 5) % 29) / 80.0), 6)),
                    "source_branch": "mechanism_generator",
                }
            )
            idx += 1
    if len(rows) < target:
        raise RuntimeError(f"mechanism generator produced {len(rows)} rows, below required target {target}")
    return pd.DataFrame(rows)
