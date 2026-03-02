"""Stage-26 context-scoped rulelet library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from buffmini.stage26.context import CONTEXTS


@dataclass(frozen=True)
class RuleletContract:
    """Contract for context-scoped rulelets."""

    name: str
    family: str
    contexts_allowed: tuple[str, ...]
    threshold: float
    default_exit: str
    _required: tuple[str, ...]
    _score_fn: Callable[[pd.DataFrame], pd.Series]
    _explain_fn: Callable[[pd.DataFrame], dict[str, Any]] = field(default=lambda _: {})

    def required_features(self) -> list[str]:
        return list(self._required)

    def compute_score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.to_numeric(self._score_fn(df), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.Series(np.clip(score.to_numpy(dtype=float), -1.0, 1.0), index=df.index, dtype=float)

    def explain(self, df_tail: pd.DataFrame) -> dict[str, Any]:
        payload = dict(self._explain_fn(df_tail))
        payload.update({"name": self.name, "family": self.family, "contexts_allowed": list(self.contexts_allowed)})
        return payload


def _ensure_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out.get("close", 0.0), errors="coerce").astype(float)
    if "atr_14" not in out.columns:
        high = pd.to_numeric(out.get("high", close), errors="coerce").astype(float)
        low = pd.to_numeric(out.get("low", close), errors="coerce").astype(float)
        tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14, min_periods=2).mean().fillna(0.0)
    if "ema_20" not in out.columns:
        out["ema_20"] = close.ewm(span=20, adjust=False).mean()
    if "ema_50" not in out.columns:
        out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    if "ema_slope_50" not in out.columns:
        out["ema_slope_50"] = (out["ema_50"] / out["ema_50"].shift(24) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "bb_mid_20" not in out.columns:
        out["bb_mid_20"] = close.rolling(20, min_periods=3).mean()
    if "bb_upper_20_2" not in out.columns or "bb_lower_20_2" not in out.columns:
        std = close.rolling(20, min_periods=3).std(ddof=0)
        out["bb_upper_20_2"] = out["bb_mid_20"] + 2.0 * std
        out["bb_lower_20_2"] = out["bb_mid_20"] - 2.0 * std
    if "donchian_high_20" not in out.columns:
        out["donchian_high_20"] = pd.to_numeric(out.get("high", close), errors="coerce").rolling(20, min_periods=2).max()
    if "donchian_low_20" not in out.columns:
        out["donchian_low_20"] = pd.to_numeric(out.get("low", close), errors="coerce").rolling(20, min_periods=2).min()
    if "ctx_state" not in out.columns:
        out["ctx_state"] = "RANGE"
    if "ctx_score_trend" not in out.columns:
        out["ctx_score_trend"] = 0.5
    if "ctx_score_range" not in out.columns:
        out["ctx_score_range"] = 0.5
    if "ctx_score_vol_expansion" not in out.columns:
        out["ctx_score_vol_expansion"] = 0.5
    if "ctx_score_vol_compression" not in out.columns:
        out["ctx_score_vol_compression"] = 0.5
    if "ctx_volume_z" not in out.columns:
        vol = pd.to_numeric(out.get("volume", 0.0), errors="coerce").fillna(0.0)
        out["ctx_volume_z"] = ((vol - vol.rolling(120, min_periods=10).mean()) / vol.rolling(120, min_periods=10).std(ddof=0).replace(0.0, np.nan)).fillna(0.0)
    return out


def build_rulelet_library() -> dict[str, RuleletContract]:
    """Build deterministic Stage-26 rulelet set (>=12)."""

    required = ("timestamp", "open", "high", "low", "close", "volume", "atr_14")

    def trend_pullback(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        close = pd.to_numeric(x["close"], errors="coerce")
        atr = pd.to_numeric(x["atr_14"], errors="coerce").replace(0.0, np.nan)
        slope = pd.to_numeric(x["ema_slope_50"], errors="coerce")
        pull = ((pd.to_numeric(x["ema_50"], errors="coerce") - close) / (atr + 1e-12)).clip(-3.0, 3.0)
        return np.tanh(slope * 100.0) * np.clip(pull, -1.5, 1.5) / 1.5

    def breakout_retest(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        close = pd.to_numeric(x["close"], errors="coerce")
        hi = pd.to_numeric(x["donchian_high_20"], errors="coerce")
        lo = pd.to_numeric(x["donchian_low_20"], errors="coerce")
        up = (close > hi.shift(1)).astype(float)
        dn = (close < lo.shift(1)).astype(float)
        retest = ((close - pd.to_numeric(x["ema_20"], errors="coerce")).abs() / (pd.to_numeric(x["atr_14"], errors="coerce") + 1e-12)).clip(0.0, 2.0)
        return (up - dn) * np.clip(1.0 - retest, 0.0, 1.0)

    def range_fade(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        close = pd.to_numeric(x["close"], errors="coerce")
        hi = pd.to_numeric(x["donchian_high_20"], errors="coerce")
        lo = pd.to_numeric(x["donchian_low_20"], errors="coerce")
        mid = (hi + lo) * 0.5
        width = (hi - lo).replace(0.0, np.nan)
        z = ((close - mid) / (width + 1e-12)).clip(-1.0, 1.0)
        return -z

    def bollinger_snapback(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        close = pd.to_numeric(x["close"], errors="coerce")
        upper = pd.to_numeric(x["bb_upper_20_2"], errors="coerce")
        lower = pd.to_numeric(x["bb_lower_20_2"], errors="coerce")
        mid = pd.to_numeric(x["bb_mid_20"], errors="coerce")
        dist = ((close - mid) / ((upper - lower).replace(0.0, np.nan) + 1e-12)).clip(-1.0, 1.0)
        return -dist

    def vol_compression_breakout(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        comp = pd.to_numeric(x["ctx_score_vol_compression"], errors="coerce").fillna(0.0)
        close = pd.to_numeric(x["close"], errors="coerce")
        brk = np.sign(close.diff().fillna(0.0))
        return comp * brk

    def vol_expansion_cont(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        exp = pd.to_numeric(x["ctx_score_vol_expansion"], errors="coerce").fillna(0.0)
        slope = np.sign(pd.to_numeric(x["ema_slope_50"], errors="coerce").fillna(0.0))
        return exp * slope

    def momentum_burst(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        volz = pd.to_numeric(x["ctx_volume_z"], errors="coerce").fillna(0.0).clip(-4.0, 4.0) / 4.0
        mom = np.sign(pd.to_numeric(x["close"], errors="coerce").diff(3).fillna(0.0))
        return volz * mom

    def mean_revert_after_spike(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        volz = pd.to_numeric(x["ctx_volume_z"], errors="coerce").fillna(0.0).clip(-4.0, 4.0) / 4.0
        dist = ((pd.to_numeric(x["close"], errors="coerce") - pd.to_numeric(x["ema_20"], errors="coerce")) / (pd.to_numeric(x["atr_14"], errors="coerce") + 1e-12)).clip(-2.0, 2.0)
        return -(np.sign(dist) * np.abs(volz))

    def structure_break(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        close = pd.to_numeric(x["close"], errors="coerce")
        brk_up = (close > pd.to_numeric(x["donchian_high_20"], errors="coerce").shift(1)).astype(float)
        brk_dn = (close < pd.to_numeric(x["donchian_low_20"], errors="coerce").shift(1)).astype(float)
        return brk_up - brk_dn

    def failed_break_reversal(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        close = pd.to_numeric(x["close"], errors="coerce")
        hi = pd.to_numeric(x["donchian_high_20"], errors="coerce")
        lo = pd.to_numeric(x["donchian_low_20"], errors="coerce")
        fail_up = (close.shift(1) > hi.shift(1)) & (close <= hi)
        fail_dn = (close.shift(1) < lo.shift(1)) & (close >= lo)
        return fail_dn.astype(float) - fail_up.astype(float)

    def chop_filter_gate(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        state = x["ctx_state"].astype(str)
        return pd.Series(np.where(state == "CHOP", 0.0, 0.5), index=x.index, dtype=float)

    def trend_flip(df: pd.DataFrame) -> pd.Series:
        x = _ensure_base(df)
        state = x["ctx_state"].astype(str)
        prev = state.shift(1).fillna("RANGE")
        flip = (prev == "CHOP") & (state == "TREND")
        slope = np.sign(pd.to_numeric(x["ema_slope_50"], errors="coerce").fillna(0.0))
        return pd.Series(np.where(flip, slope, 0.0), index=x.index, dtype=float)

    entries: list[RuleletContract] = [
        RuleletContract("TrendPullback", "price", ("TREND",), 0.25, "fixed_atr", required, trend_pullback),
        RuleletContract("BreakoutRetest", "price", ("TREND", "VOL_EXPANSION"), 0.30, "atr_trailing", required, breakout_retest),
        RuleletContract("RangeFade", "price", ("RANGE",), 0.25, "fixed_atr", required, range_fade),
        RuleletContract("BollingerSnapBack", "price", ("RANGE",), 0.30, "fixed_atr", required, bollinger_snapback),
        RuleletContract("VolCompressionBreakout", "volatility", ("VOL_COMPRESSION",), 0.30, "atr_trailing", required, vol_compression_breakout),
        RuleletContract("VolExpansionContinuation", "volatility", ("VOL_EXPANSION",), 0.30, "atr_trailing", required, vol_expansion_cont),
        RuleletContract("MomentumBurst", "flow", ("VOLUME_SHOCK", "TREND"), 0.25, "fixed_atr", required, momentum_burst),
        RuleletContract("MeanRevertAfterSpike", "flow", ("VOLUME_SHOCK", "RANGE"), 0.25, "fixed_atr", required, mean_revert_after_spike),
        RuleletContract("StructureBreak", "price", ("TREND",), 0.30, "atr_trailing", required, structure_break),
        RuleletContract("FailedBreakReversal", "price", ("RANGE", "VOL_EXPANSION"), 0.25, "fixed_atr", required, failed_break_reversal),
        RuleletContract("ChopFilterGate", "risk", ("CHOP",), 0.10, "fixed_atr", required, chop_filter_gate),
        RuleletContract("TrendFlip", "price", ("CHOP", "TREND"), 0.20, "fixed_atr", required, trend_flip),
    ]
    return {item.name: item for item in entries}


def available_contexts() -> list[str]:
    return list(CONTEXTS)

