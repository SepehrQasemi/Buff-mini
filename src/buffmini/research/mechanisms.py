"""Mechanism-based hypothesis generation for discovery and campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class MechanismSpec:
    family: str
    subfamilies: tuple[str, ...]
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
    expected_regimes: tuple[str, ...]
    expected_failure_modes: tuple[str, ...]
    trade_density_expectation: str
    transfer_expectation: str
    transfer_risk_prior: float


MECHANISM_SPECS: tuple[MechanismSpec, ...] = (
    MechanismSpec(
        family="structure_pullback_continuation",
        subfamilies=("shallow_pullback", "deep_pullback", "micro_break_rejoin", "volatility_reset_reclaim"),
        contexts=("trend", "transition", "compression"),
        triggers=("pullback_reclaim", "micro_break_reclaim", "htf_bias_reclaim"),
        confirmations=("volume_expansion_confirm", "flow_continuation_confirm", "structure_repair_confirm"),
        participation_styles=("aligned_flow", "break_reclaim", "impulse_rejoin", "retest_then_continue"),
        invalidations=("structure_break", "failed_reclaim", "impulse_failure"),
        risk_models=("trend_pullback",),
        exit_families=("scale_out_then_trail", "trailing_atr", "partial_then_time"),
        time_stops=(8, 12, 16, 20),
        session_filters=("any_session", "ldn_ny_overlap"),
        modules=("structure_engine", "flow_regime_engine", "mtf_bias_completion"),
        expected_regimes=("trend", "transition"),
        expected_failure_modes=("late_trend", "shallow_follow_through", "transfer_decay", "retest_failure"),
        trade_density_expectation="medium",
        transfer_expectation="moderate",
        transfer_risk_prior=0.22,
    ),
    MechanismSpec(
        family="liquidity_sweep_reversal",
        subfamilies=("session_sweep", "range_sweep", "pool_reclaim", "double_sweep_reclaim"),
        contexts=("range", "transition", "liquidity_rotation"),
        triggers=("liquidity_sweep_reclaim", "pool_violation_reclaim", "stop_run_reclaim"),
        confirmations=("rejection_close_confirm", "flow_exhaustion_confirm", "delta_reversal_confirm"),
        participation_styles=("wick_rejection", "close_back_inside", "delayed_reclaim", "reclaim_after_flush"),
        invalidations=("failed_reclaim", "flow_reversal", "second_push_break"),
        risk_models=("sweep_reversal",),
        exit_families=("fixed_rr", "breakeven_then_fixed", "time_exit"),
        time_stops=(4, 6, 8, 10),
        session_filters=("any_session", "session_extremes"),
        modules=("liquidity_map", "flow_regime_engine", "trade_geometry_layer"),
        expected_regimes=("range", "transition"),
        expected_failure_modes=("trend_resume", "wick_only_reversal", "late_reclaim", "liquidity_retest_fail"),
        trade_density_expectation="medium",
        transfer_expectation="moderate",
        transfer_risk_prior=0.28,
    ),
    MechanismSpec(
        family="squeeze_flow_breakout",
        subfamilies=("opening_squeeze", "midrange_compression", "retest_breakout", "nested_compression_break"),
        contexts=("vol_shock", "transition", "compression"),
        triggers=("compression_breakout", "range_expansion_break", "squeeze_release_retest"),
        confirmations=("volume_expansion_confirm", "flow_burst_confirm", "participation_cluster_confirm"),
        participation_styles=("breakout_chase", "retest_hold", "opening_drive", "staged_breakout"),
        invalidations=("volatility_recompression", "failed_breakout", "range_reacceptance"),
        risk_models=("breakout_follow_through",),
        exit_families=("trailing_atr", "scale_out_then_trail", "partial_then_time"),
        time_stops=(10, 16, 24, 32),
        session_filters=("any_session", "high_liquidity_hours", "session_opening_block"),
        modules=("volatility_regime_engine", "flow_regime_engine", "mtf_bias_completion"),
        expected_regimes=("compression", "high_vol"),
        expected_failure_modes=("false_break", "late_expansion", "cost_fragility", "reacceptance_chop"),
        trade_density_expectation="low",
        transfer_expectation="moderate",
        transfer_risk_prior=0.24,
    ),
    MechanismSpec(
        family="failed_breakout_reversal",
        subfamilies=("inside_fake_break", "session_extreme_fail", "trend_exhaust_fail", "liquidity_void_fail"),
        contexts=("transition", "vol_shock", "liquidity_rotation"),
        triggers=("failed_breakout_snapback", "fake_breakout_reversal", "breakout_failure_flush"),
        confirmations=("exhaustion_confirm", "reclaim_confirm", "auction_reject_confirm"),
        participation_styles=("snapback_entry", "close_back_inside", "reclaim_retest", "flush_then_reclaim"),
        invalidations=("breakout_reacceptance", "trend_continuation", "second_range_accept"),
        risk_models=("failed_breakout_reversal",),
        exit_families=("fixed_rr", "breakeven_then_fixed", "time_exit"),
        time_stops=(4, 6, 10, 12),
        session_filters=("any_session", "session_extremes"),
        modules=("liquidity_map", "volatility_regime_engine", "flow_regime_engine"),
        expected_regimes=("transition", "high_vol"),
        expected_failure_modes=("trend_resume", "double_fake_break", "thin_follow_through", "reclaim_delay"),
        trade_density_expectation="low",
        transfer_expectation="low",
        transfer_risk_prior=0.35,
    ),
    MechanismSpec(
        family="exhaustion_mean_reversion",
        subfamilies=("atr_stretch_fade", "band_snapback", "panic_mean_revert", "basis_snapback"),
        contexts=("range", "vol_shock", "funding_extreme"),
        triggers=("stretch_extreme_revert", "exhaustion_reversal", "basis_dislocation_revert"),
        confirmations=("flow_exhaustion_confirm", "volatility_fade_confirm", "reversion_velocity_confirm"),
        participation_styles=("fade_extreme", "close_reclaim", "mean_snap", "staggered_fade"),
        invalidations=("trend_resume", "new_extreme_break", "mean_drift_failure"),
        risk_models=("mean_reversion",),
        exit_families=("fixed_rr", "time_exit"),
        time_stops=(4, 8, 12, 16),
        session_filters=("any_session", "late_session"),
        modules=("flow_regime_engine", "trade_geometry_layer"),
        expected_regimes=("range", "high_vol", "funding_extreme"),
        expected_failure_modes=("one_way_trend", "late_mean_revert", "cost_decay", "crowding_persistence"),
        trade_density_expectation="medium",
        transfer_expectation="low",
        transfer_risk_prior=0.40,
    ),
    MechanismSpec(
        family="funding_oi_imbalance_reversion",
        subfamilies=("funding_flush", "oi_divergence", "crowding_release", "basis_dislocation_revert"),
        contexts=("crowding", "transition"),
        triggers=("funding_flush_reversion", "oi_divergence_revert", "basis_dislocation_revert"),
        confirmations=("crowding_extreme_confirm", "sentiment_reversal_confirm", "open_interest_rollover_confirm"),
        participation_styles=("sentiment_snap", "imbalance_fade", "delayed_reversal", "basis_reclaim"),
        invalidations=("crowding_acceleration", "trend_resume", "basis_continuation"),
        risk_models=("crowding_reversion",),
        exit_families=("fixed_rr", "time_exit"),
        time_stops=(6, 12, 18, 24),
        session_filters=("any_session", "high_liquidity_hours"),
        modules=("crowding_layer", "flow_regime_engine"),
        expected_regimes=("funding_extreme", "crowding", "transition"),
        expected_failure_modes=("crowding_persistence", "opportunity_sparsity", "transfer_decay", "timing_instability"),
        trade_density_expectation="low",
        transfer_expectation="low",
        transfer_risk_prior=0.48,
    ),
    MechanismSpec(
        family="volatility_regime_transition",
        subfamilies=("compression_release", "vol_reclaim", "atr_regime_shift", "quiet_trend_release"),
        contexts=("transition", "vol_shock", "compression"),
        triggers=("compression_to_expansion", "vol_regime_reclaim", "vol_pivot_break"),
        confirmations=("breakout_readiness_confirm", "momentum_slope_confirm", "range_escape_confirm"),
        participation_styles=("anticipatory_break", "confirmation_break", "retest_break", "late_escape_retest"),
        invalidations=("recompression", "failed_expansion", "expansion_fade"),
        risk_models=("vol_transition",),
        exit_families=("trailing_atr", "time_exit", "scale_out_then_trail"),
        time_stops=(8, 14, 20, 28),
        session_filters=("any_session", "high_liquidity_hours"),
        modules=("volatility_regime_engine", "structure_engine"),
        expected_regimes=("compression", "transition"),
        expected_failure_modes=("recompression", "late_breakout", "timing_instability", "expansion_exhaustion"),
        trade_density_expectation="medium",
        transfer_expectation="moderate",
        transfer_risk_prior=0.30,
    ),
    MechanismSpec(
        family="multi_tf_disagreement_repair",
        subfamilies=("htf_lag_repair", "ltf_flip_realign", "bias_reclaim", "compression_bias_realign"),
        contexts=("transition", "trend", "range"),
        triggers=("htf_ltf_disagreement_repair", "bias_realignment_break", "compression_bias_realign"),
        confirmations=("structure_realign_confirm", "flow_continuation_confirm", "bias_reaccept_confirm"),
        participation_styles=("alignment_reclaim", "late_confirmation", "pullback_after_realign", "drift_repair_entry"),
        invalidations=("misalignment_persists", "htf_bias_flip", "alignment_reject"),
        risk_models=("multi_tf_repair",),
        exit_families=("trailing_atr", "scale_out_then_trail", "partial_then_time"),
        time_stops=(8, 12, 18, 24),
        session_filters=("any_session", "ldn_ny_overlap"),
        modules=("mtf_bias_completion", "structure_engine", "flow_regime_engine"),
        expected_regimes=("trend", "transition", "compression"),
        expected_failure_modes=("misalignment_persists", "late_confirmation", "transfer_decay", "bias_flipback"),
        trade_density_expectation="medium",
        transfer_expectation="moderate",
        transfer_risk_prior=0.26,
    ),
)


def mechanism_families() -> tuple[str, ...]:
    return tuple(spec.family for spec in MECHANISM_SPECS)


def mechanism_registry() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in MECHANISM_SPECS:
        rows.append(
            {
                "family": spec.family,
                "subfamily_count": len(spec.subfamilies),
                "subfamilies": list(spec.subfamilies),
                "contexts": list(spec.contexts),
                "triggers": list(spec.triggers),
                "confirmations": list(spec.confirmations),
                "participation_styles": list(spec.participation_styles),
                "invalidations": list(spec.invalidations),
                "risk_models": list(spec.risk_models),
                "exit_families": list(spec.exit_families),
                "time_stops": list(spec.time_stops),
                "session_filters": list(spec.session_filters),
                "modules": list(spec.modules),
                "expected_regimes": list(spec.expected_regimes),
                "expected_failure_modes": list(spec.expected_failure_modes),
                "trade_density_expectation": spec.trade_density_expectation,
                "transfer_expectation": spec.transfer_expectation,
                "transfer_risk_prior": float(spec.transfer_risk_prior),
            }
        )
    return rows


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
            subfamily = spec.subfamilies[idx % max(1, len(spec.subfamilies))]
            expected_regime = spec.expected_regimes[idx % max(1, len(spec.expected_regimes))]
            mechanism_signature = stable_hash(
                {
                    "family": spec.family,
                    "subfamily": subfamily,
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
                    "expected_regime": expected_regime,
                },
                length=20,
            )
            rows.append(
                {
                    "candidate_id": f"s70_{stable_hash({'i': idx, 'fp': mechanism_signature}, length=16)}",
                    "family": spec.family,
                    "subfamily": subfamily,
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
                    "expected_regime": expected_regime,
                    "expected_failure_modes": list(spec.expected_failure_modes),
                    "trade_density_expectation": spec.trade_density_expectation,
                    "transfer_expectation": spec.transfer_expectation,
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
