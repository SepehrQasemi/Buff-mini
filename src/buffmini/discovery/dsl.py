"""Stage-9 DSL-lite regime selectors for discovery.

This module intentionally provides only minimal primitives that *select between*
entry families. It never hard-blocks trading.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


FUNDING_POS_EXTREME = "POS_EXTREME"
FUNDING_NEG_EXTREME = "NEG_EXTREME"
FUNDING_NEUTRAL = "NEUTRAL"

OI_BUILDING = "BUILDING"
OI_UNWINDING = "UNWINDING"
OI_NEUTRAL = "NEUTRAL"


def funding_regime(frame: pd.DataFrame) -> pd.Series:
    """Classify funding regime from Stage-9 funding extreme features."""

    pos = pd.to_numeric(frame.get("funding_extreme_pos", 0), errors="coerce").fillna(0) > 0
    neg = pd.to_numeric(frame.get("funding_extreme_neg", 0), errors="coerce").fillna(0) > 0

    regime = pd.Series(FUNDING_NEUTRAL, index=frame.index, dtype="object")
    regime.loc[neg] = FUNDING_NEG_EXTREME
    regime.loc[pos] = FUNDING_POS_EXTREME
    return regime


def oi_regime(frame: pd.DataFrame) -> pd.Series:
    """Classify open-interest regime from Stage-9 OI change features."""

    chg_24 = pd.to_numeric(frame.get("oi_chg_24", 0.0), errors="coerce").fillna(0.0)
    regime = pd.Series(OI_NEUTRAL, index=frame.index, dtype="object")
    regime.loc[chg_24 > 0.0] = OI_BUILDING
    regime.loc[chg_24 < 0.0] = OI_UNWINDING
    return regime


def select_signals_by_regime(
    frame: pd.DataFrame,
    primary_signal: pd.Series,
    alternate_signal: pd.Series,
    *,
    use_funding_selector: bool = True,
    use_oi_selector: bool = True,
) -> pd.Series:
    """Select between two signal families based on funding/OI regimes.

    The selector is non-blocking by design: it only switches when both families
    currently emit non-zero signals. Otherwise it preserves the primary signal.
    """

    primary = pd.Series(primary_signal, index=frame.index).fillna(0).astype(int)
    alternate = pd.Series(alternate_signal, index=frame.index).fillna(0).astype(int)

    switch_mask = pd.Series(False, index=frame.index)
    if use_funding_selector:
        f_regime = funding_regime(frame)
        switch_mask = switch_mask | f_regime.isin({FUNDING_POS_EXTREME, FUNDING_NEG_EXTREME})
    if use_oi_selector:
        o_regime = oi_regime(frame)
        switch_mask = switch_mask | (o_regime == OI_BUILDING)

    # Non-blocking switch: only swap when both families are actively signaling.
    safe_switch = switch_mask & primary.ne(0) & alternate.ne(0)
    selected = primary.copy()
    selected.loc[safe_switch] = alternate.loc[safe_switch]
    return selected.astype(int)


def dsl_lite_enabled(config: dict[str, Any] | None) -> bool:
    """Return whether Stage-9 DSL-lite selector is enabled."""

    if not isinstance(config, dict):
        return False
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return False
    stage9 = evaluation.get("stage9", {})
    if not isinstance(stage9, dict):
        return False
    dsl_lite = stage9.get("dsl_lite", {})
    if not isinstance(dsl_lite, dict):
        return False
    return bool(dsl_lite.get("enabled", False))


def dsl_lite_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized Stage-9 DSL-lite settings."""

    defaults = {
        "enabled": False,
        "funding_selector_enabled": True,
        "oi_selector_enabled": True,
    }
    if not isinstance(config, dict):
        return defaults
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return defaults
    stage9 = evaluation.get("stage9", {})
    if not isinstance(stage9, dict):
        return defaults
    dsl_lite = stage9.get("dsl_lite", {})
    if not isinstance(dsl_lite, dict):
        return defaults

    merged = dict(defaults)
    merged.update({k: v for k, v in dsl_lite.items() if k in merged})
    return merged

