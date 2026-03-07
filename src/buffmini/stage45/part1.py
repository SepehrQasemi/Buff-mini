"""Stage-45 analyst brain part 1 deterministic modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage44.contracts import (
    build_allocator_hook,
    build_contribution_record,
    build_failure_record,
    build_runtime_event,
    to_registry_row,
)


def _numeric(frame: pd.DataFrame, key: str, fallback: float = 0.0) -> pd.Series:
    if key not in frame.columns:
        return pd.Series(fallback, index=frame.index, dtype=float)
    return pd.to_numeric(frame[key], errors="coerce").fillna(fallback).astype(float)


def compute_market_structure_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic structure states (HH/HL/LH/LL/BOS/CHOCH)."""

    data = frame.copy()
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    close = _numeric(data, "close")
    prev_high_3 = high.shift(1).rolling(3, min_periods=1).max()
    prev_low_3 = low.shift(1).rolling(3, min_periods=1).min()

    hh = (high > prev_high_3).fillna(False)
    hl = (low > prev_low_3).fillna(False)
    lh = (high < prev_high_3).fillna(False)
    ll = (low < prev_low_3).fillna(False)
    bos_up = (close > prev_high_3).fillna(False)
    bos_down = (close < prev_low_3).fillna(False)
    bos = bos_up | bos_down

    struct_sign = pd.Series(np.where(bos_up, 1, np.where(bos_down, -1, 0)), index=data.index, dtype=int)
    choch = (struct_sign != 0) & (struct_sign.shift(1).fillna(0) != 0) & (struct_sign != struct_sign.shift(1))

    ret = close.pct_change().fillna(0.0)
    impulse_threshold = ret.abs().rolling(30, min_periods=1).median() * 1.5
    impulsive = (ret.abs() >= impulse_threshold).fillna(False)
    corrective = (~impulsive).fillna(True)

    bias_score = close.diff().rolling(48, min_periods=1).sum().fillna(0.0)
    structural_bias = pd.Series(
        np.where(bias_score > 0, "bull", np.where(bias_score < 0, "bear", "range")),
        index=data.index,
        dtype=str,
    )

    return pd.DataFrame(
        {
            "higher_high": hh.astype(bool),
            "higher_low": hl.astype(bool),
            "lower_high": lh.astype(bool),
            "lower_low": ll.astype(bool),
            "bos": bos.astype(bool),
            "choch": choch.astype(bool),
            "impulsive_leg": impulsive.astype(bool),
            "corrective_leg": corrective.astype(bool),
            "structural_bias": structural_bias,
        },
        index=data.index,
    )


def compute_liquidity_map(frame: pd.DataFrame, *, tolerance_ratio: float = 0.0005) -> pd.DataFrame:
    """Compute deterministic liquidity pools/sweeps/fake-breakouts."""

    data = frame.copy()
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    close = _numeric(data, "close")

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    equal_high = ((high - prev_high).abs() / prev_high.replace(0, np.nan)).fillna(np.inf) <= float(tolerance_ratio)
    equal_low = ((low - prev_low).abs() / prev_low.replace(0, np.nan)).fillna(np.inf) <= float(tolerance_ratio)

    pool_high = equal_high.rolling(5, min_periods=1).sum() >= 2
    pool_low = equal_low.rolling(5, min_periods=1).sum() >= 2
    local_high = high.shift(1).rolling(12, min_periods=1).max()
    local_low = low.shift(1).rolling(12, min_periods=1).min()
    sweep_high = (high > local_high) & (close < local_high)
    sweep_low = (low < local_low) & (close > local_low)
    fake_breakout = sweep_high | sweep_low

    return pd.DataFrame(
        {
            "equal_highs": equal_high.astype(bool),
            "equal_lows": equal_low.astype(bool),
            "liquidity_pool_high": pool_high.astype(bool),
            "liquidity_pool_low": pool_low.astype(bool),
            "liquidity_sweep_high": sweep_high.fillna(False).astype(bool),
            "liquidity_sweep_low": sweep_low.fillna(False).astype(bool),
            "fake_breakout": fake_breakout.fillna(False).astype(bool),
        },
        index=data.index,
    )


def compute_volatility_regime_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic compression/expansion/clustering/squeeze states."""

    data = frame.copy()
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    close = _numeric(data, "close")
    tr = (high - low).abs()
    vol = close.pct_change().rolling(24, min_periods=1).std().fillna(0.0)
    low_q = vol.rolling(120, min_periods=10).quantile(0.25).fillna(vol.median())
    high_q = vol.rolling(120, min_periods=10).quantile(0.75).fillna(vol.median())
    compression = (vol <= low_q).fillna(False)
    expansion = (vol >= high_q).fillna(False)
    clustering = (vol.diff().abs() <= vol.rolling(12, min_periods=1).median().fillna(0.0)).fillna(False)
    squeeze_state = compression & (tr <= tr.rolling(24, min_periods=1).median().fillna(0.0))
    breakout_readiness = (squeeze_state.shift(1).fillna(False) & (vol.diff().fillna(0.0) > 0.0)).fillna(False)

    return pd.DataFrame(
        {
            "volatility_compression": compression.astype(bool),
            "volatility_expansion": expansion.astype(bool),
            "volatility_clustering": clustering.astype(bool),
            "squeeze_state": squeeze_state.astype(bool),
            "breakout_readiness": breakout_readiness.astype(bool),
        },
        index=data.index,
    )


def compute_htf_bias_skeleton(frame: pd.DataFrame, *, factor: int = 4) -> pd.DataFrame:
    """Compute deterministic higher-timeframe directional/range/stress bias."""

    data = frame.copy()
    close = _numeric(data, "close")
    vol = close.pct_change().rolling(24, min_periods=1).std().fillna(0.0)
    group = (np.arange(len(data)) // max(1, int(factor))).astype(int)
    htf_close = close.groupby(group).last()
    ma_fast = htf_close.rolling(4, min_periods=1).mean()
    ma_slow = htf_close.rolling(8, min_periods=1).mean()
    htf_dir = pd.Series(np.where(ma_fast > ma_slow, "up", np.where(ma_fast < ma_slow, "down", "flat")), index=htf_close.index, dtype=str)
    htf_range = (htf_close.pct_change().abs().rolling(6, min_periods=1).mean().fillna(0.0) < 0.003)
    htf_stress = (vol.groupby(group).mean() >= vol.rolling(120, min_periods=10).quantile(0.75).fillna(vol.median()).groupby(group).last()).fillna(False)

    out = pd.DataFrame(
        {
            "htf_directional_bias": htf_dir,
            "htf_range_bias": htf_range.astype(bool),
            "htf_stress_regime": htf_stress.astype(bool),
        }
    ).reindex(group).reset_index(drop=True)
    out.index = data.index
    return out


def build_stage45_contract_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Create Stage-44-compatible records for Stage-45 modules."""

    structure = compute_market_structure_engine(frame)
    liquidity = compute_liquidity_map(frame)
    vol = compute_volatility_regime_engine(frame)
    htf = compute_htf_bias_skeleton(frame)
    modules = {
        "structure_engine": structure,
        "liquidity_map": liquidity,
        "volatility_regime_engine": vol,
        "htf_bias_skeleton": htf,
    }

    rows: list[dict[str, Any]] = []
    for module_name, module_df in modules.items():
        active_ratio = float(module_df.select_dtypes(include=["bool"]).mean().mean()) if not module_df.empty else 0.0
        contribution = build_contribution_record(
            module_name=module_name,
            family_name="analyst_brain_part1",
            setup_name=f"{module_name}_state",
            raw_candidate_contribution=float(active_ratio),
            stage_a_survival_lift=float(active_ratio * 0.6),
            stage_b_survival_lift=float(active_ratio * 0.4),
            final_policy_contribution=float(active_ratio * 0.3),
            runtime_seconds=0.001,
            registry_rows_added=1,
            cost_of_use_if_measurable=None,
            coverage_flags={"states_emitted": bool(module_df.shape[0] > 0)},
        )
        failure = build_failure_record(
            module_name=module_name,
            family_name="analyst_brain_part1",
            motif="REJECT::NO_STRUCTURE_CONFIRMATION" if module_name == "structure_engine" else "REJECT::NO_SIGNAL",
            details={"active_ratio": active_ratio},
        )
        runtime = build_runtime_event(
            module_name=module_name,
            phase_name=f"{module_name}_phase",
            enter_ts=1.0,
            exit_ts=1.002,
            candidate_rows_in=int(frame.shape[0]),
            candidate_rows_out=int(module_df.shape[0]),
        )
        allocator = build_allocator_hook(
            module_name=module_name,
            family_name="analyst_brain_part1",
            exploration_eligible=True,
            exploitation_score=float(active_ratio),
            uncertainty_score=float(max(0.0, 1.0 - active_ratio)),
            novelty_score=0.2,
            min_exploration_floor=0.1,
        )
        row = to_registry_row(
            module_name=module_name,
            family_name="analyst_brain_part1",
            setup_name=f"{module_name}_state",
            contribution_summary=contribution,
            failure_motifs=[str(failure["motif"])],
            runtime_metrics=runtime,
            allocator_hook=allocator,
            mutation_guidance="expand_context_with_flow_confirmation",
        )
        rows.append(row)
    return rows

