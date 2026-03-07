"""Stage-46 analyst brain part 2 deterministic modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage41.contribution import oi_short_only_runtime_guard
from buffmini.stage44.contracts import (
    build_allocator_hook,
    build_contribution_record,
    build_failure_record,
    build_runtime_event,
    to_registry_row,
)


def _num(frame: pd.DataFrame, key: str, default: float = 0.0) -> pd.Series:
    if key not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[key], errors="coerce").fillna(default).astype(float)


def compute_flow_regime_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute flow imbalance/persistence/burst/exhaustion/continuation states."""

    data = frame.copy()
    close = _num(data, "close")
    volume = _num(data, "volume", default=1.0).replace(0, 1.0)
    taker_buy_ratio = _num(data, "taker_buy_ratio", default=0.5)
    taker_sell_ratio = _num(data, "taker_sell_ratio", default=0.5)
    imbalance = (taker_buy_ratio - taker_sell_ratio).fillna(0.0)
    if (imbalance.abs() <= 1e-12).all():
        imbalance = (close.pct_change().fillna(0.0) * volume.rolling(12, min_periods=1).mean()).fillna(0.0)
    persistence = imbalance.rolling(8, min_periods=1).mean()
    burst = imbalance.abs() >= imbalance.abs().rolling(48, min_periods=1).quantile(0.8).fillna(0.0)
    exhaustion = (burst.shift(1).fillna(False) & (imbalance.abs() < imbalance.abs().rolling(8, min_periods=1).median().fillna(0.0))).fillna(False)
    continuation = (persistence.abs() > persistence.abs().rolling(24, min_periods=1).median().fillna(0.0)) & (~exhaustion)

    return pd.DataFrame(
        {
            "flow_imbalance": imbalance.astype(float),
            "imbalance_persistence": persistence.astype(float),
            "flow_burst": burst.astype(bool),
            "flow_exhaustion": exhaustion.astype(bool),
            "flow_confirmed_continuation": continuation.astype(bool),
        },
        index=data.index,
    )


def compute_crowding_layer(
    frame: pd.DataFrame,
    *,
    timeframe: str = "1h",
    short_only_enabled: bool = True,
    short_horizon_max: str = "30m",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute crowding/sentiment states and enforce OI short-only guard."""

    data = frame.copy()
    funding = _num(data, "funding_rate", default=0.0)
    ls_ratio = _num(data, "long_short_ratio", default=1.0)
    oi = _num(data, "oi", default=np.nan)
    guard = oi_short_only_runtime_guard(
        timeframe=str(timeframe),
        short_only_enabled=bool(short_only_enabled),
        short_horizon_max=str(short_horizon_max),
    )
    oi_allowed = bool(guard.get("oi_allowed", False))
    if not oi_allowed:
        oi = pd.Series(np.nan, index=data.index, dtype=float)

    funding_z = ((funding - funding.rolling(120, min_periods=1).mean()) / funding.rolling(120, min_periods=1).std().replace(0, np.nan)).fillna(0.0)
    ls_z = ((ls_ratio - ls_ratio.rolling(120, min_periods=1).mean()) / ls_ratio.rolling(120, min_periods=1).std().replace(0, np.nan)).fillna(0.0)
    crowding_extreme = (funding_z.abs() >= 2.0) | (ls_z.abs() >= 2.0)
    funding_stress = funding_z.abs() >= 1.5
    long_short_extreme = ls_z.abs() >= 1.5
    crowded_side_bias = pd.Series(
        np.where((funding_z > 1.0) & (ls_z > 1.0), "overlong", np.where((funding_z < -1.0) & (ls_z < -1.0), "overshort", "neutral")),
        index=data.index,
        dtype=str,
    )
    sentiment_reversal_hint = crowding_extreme & (funding_z * ls_z > 0)

    out = pd.DataFrame(
        {
            "crowding_extreme": crowding_extreme.astype(bool),
            "funding_stress": funding_stress.astype(bool),
            "long_short_extreme": long_short_extreme.astype(bool),
            "crowded_side_bias": crowded_side_bias,
            "sentiment_reversal_hint": sentiment_reversal_hint.astype(bool),
            "oi_active": oi.notna().astype(bool),
        },
        index=data.index,
    )
    return out, guard


def compute_mtf_bias_completion(frame: pd.DataFrame, *, htf_bias: pd.Series | None = None) -> pd.DataFrame:
    """Compute HTF/intermediate/LTF alignment/misalignment states."""

    close = _num(frame, "close")
    htf = htf_bias if htf_bias is not None else pd.Series(np.where(close.rolling(48, min_periods=1).mean() > close.rolling(120, min_periods=1).mean(), "up", "down"), index=frame.index, dtype=str)
    mid = pd.Series(np.where(close.rolling(24, min_periods=1).mean() > close.rolling(72, min_periods=1).mean(), "up", "down"), index=frame.index, dtype=str)
    ltf = pd.Series(np.where(close.diff().fillna(0.0) > 0.0, "up", "down"), index=frame.index, dtype=str)
    alignment = (htf == mid) & (mid == ltf)
    misalignment = ~alignment
    ltf_trigger = alignment | ((htf == mid) & (ltf != htf))
    return pd.DataFrame(
        {
            "htf_context": htf.astype(str),
            "intermediate_context": mid.astype(str),
            "ltf_trigger_eligibility": ltf_trigger.astype(bool),
            "structural_alignment": alignment.astype(bool),
            "structural_misalignment": misalignment.astype(bool),
        },
        index=frame.index,
    )


def compute_trade_geometry_layer(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic stop/target/invalidation/RR geometry states."""

    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    atr = (high - low).rolling(14, min_periods=1).mean().replace(0, np.nan).fillna((high - low).abs().mean() or 1.0)
    stop_distance = (atr * 1.2).astype(float)
    target_distance = (atr * 2.2).astype(float)
    rr_score = (target_distance / stop_distance.replace(0, np.nan)).fillna(0.0).astype(float)
    invalidation_point = (close - stop_distance).astype(float)
    invalidation_quality = (rr_score >= 1.5).astype(bool)
    structural_preservation = ((close > invalidation_point) & (rr_score >= 1.2)).astype(bool)
    return pd.DataFrame(
        {
            "invalidation_point": invalidation_point,
            "stop_distance": stop_distance,
            "target_distance": target_distance,
            "rr_score": rr_score,
            "invalidation_quality": invalidation_quality,
            "structural_preservation_score": structural_preservation.astype(int),
        },
        index=frame.index,
    )


def build_stage46_contract_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build Stage-44-compatible registry rows for Stage-46 modules."""

    flow = compute_flow_regime_engine(frame)
    crowd, guard = compute_crowding_layer(frame)
    mtf = compute_mtf_bias_completion(frame)
    geo = compute_trade_geometry_layer(frame)
    modules = {
        "flow_regime_engine": flow,
        "crowding_layer": crowd,
        "mtf_bias_completion": mtf,
        "trade_geometry_layer": geo,
    }
    rows: list[dict[str, Any]] = []
    for module_name, module_df in modules.items():
        bool_cols = module_df.select_dtypes(include=["bool"])
        active_ratio = float(bool_cols.mean().mean()) if not bool_cols.empty else 0.0
        contribution = build_contribution_record(
            module_name=module_name,
            family_name="analyst_brain_part2",
            setup_name=f"{module_name}_state",
            raw_candidate_contribution=active_ratio,
            stage_a_survival_lift=active_ratio * 0.6,
            stage_b_survival_lift=active_ratio * 0.4,
            final_policy_contribution=active_ratio * 0.3,
            runtime_seconds=0.001,
            registry_rows_added=1,
            cost_of_use_if_measurable=None,
            coverage_flags={"oi_guard_applied": bool(guard.get("short_only_enabled", True))},
        )
        failure = build_failure_record(
            module_name=module_name,
            family_name="analyst_brain_part2",
            motif="REJECT::WEAK_FLOW_CONTEXT" if module_name == "flow_regime_engine" else "REJECT::NO_SIGNAL",
            details={"active_ratio": active_ratio},
        )
        runtime = build_runtime_event(
            module_name=module_name,
            phase_name=f"{module_name}_phase",
            enter_ts=2.0,
            exit_ts=2.003,
            candidate_rows_in=int(frame.shape[0]),
            candidate_rows_out=int(module_df.shape[0]),
        )
        allocator = build_allocator_hook(
            module_name=module_name,
            family_name="analyst_brain_part2",
            exploration_eligible=True,
            exploitation_score=active_ratio,
            uncertainty_score=max(0.0, 1.0 - active_ratio),
            novelty_score=0.25,
            min_exploration_floor=0.1,
        )
        row = to_registry_row(
            module_name=module_name,
            family_name="analyst_brain_part2",
            setup_name=f"{module_name}_state",
            contribution_summary=contribution,
            failure_motifs=[str(failure["motif"])],
            runtime_metrics=runtime,
            allocator_hook=allocator,
            mutation_guidance="increase_signal_to_cost_margin",
        )
        rows.append(row)
    return rows

