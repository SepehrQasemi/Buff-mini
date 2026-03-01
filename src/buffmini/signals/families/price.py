"""Stage-13.2 price-structure family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.signals.family_base import FamilyContext, SignalFamily


class PriceStructureFamily(SignalFamily):
    """Score-based price structure family (no hard AND-chain gates)."""

    name = "price"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def required_features(self) -> list[str]:
        return [
            "timestamp",
            "close",
            "high",
            "low",
            "atr_14",
            "ema_20",
            "ema_50",
            "ema_slope_50",
            "donchian_high_20",
            "donchian_low_20",
            "atr_pct_rank_252",
            "score_trend",
            "score_range",
        ]

    def compute_scores(self, df: pd.DataFrame, ctx: FamilyContext) -> pd.Series:
        self.validate_frame(df)
        p = {
            "donchian_period": int(self.params.get("donchian_period", 20)),
            "retest_bars": int(self.params.get("retest_bars", 6)),
            "retest_atr_k": float(self.params.get("retest_atr_k", 0.8)),
            "trend_pullback_k": float(self.params.get("trend_pullback_k", 1.2)),
            "false_break_lookback": int(self.params.get("false_break_lookback", 4)),
            "seq_min_len": int(self.params.get("seq_min_len", 2)),
        }

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        atr = pd.to_numeric(df["atr_14"], errors="coerce").replace(0.0, np.nan).astype(float)
        ema20 = pd.to_numeric(df["ema_20"], errors="coerce").astype(float)
        ema50 = pd.to_numeric(df["ema_50"], errors="coerce").astype(float)
        slope = pd.to_numeric(df["ema_slope_50"], errors="coerce").astype(float)
        high_n = pd.to_numeric(df[f"donchian_high_{p['donchian_period']}"], errors="coerce").astype(float)
        low_n = pd.to_numeric(df[f"donchian_low_{p['donchian_period']}"], errors="coerce").astype(float)
        atr_rank = pd.to_numeric(df["atr_pct_rank_252"], errors="coerce").fillna(0.5).astype(float)
        score_trend = pd.to_numeric(df["score_trend"], errors="coerce").fillna(0.0).astype(float)
        score_range = pd.to_numeric(df["score_range"], errors="coerce").fillna(0.0).astype(float)

        breakout_up = close > high_n
        breakout_dn = close < low_n
        recent_break_up = breakout_up.rolling(window=p["retest_bars"], min_periods=1).max().shift(1).fillna(0).astype(bool)
        recent_break_dn = breakout_dn.rolling(window=p["retest_bars"], min_periods=1).max().shift(1).fillna(0).astype(bool)
        retest_to_mean = ((close - ema20).abs() <= (p["retest_atr_k"] * (atr + 1e-12))).fillna(False)
        rejection_up = close > close.shift(1)
        rejection_dn = close < close.shift(1)
        breakout_retest = (
            (recent_break_up & retest_to_mean & rejection_up).astype(float)
            - (recent_break_dn & retest_to_mean & rejection_dn).astype(float)
        )

        pullback_up = (ema50 - close) / (atr + 1e-12)
        pullback_dn = (close - ema50) / (atr + 1e-12)
        trend_pullback = (
            ((slope > 0) & (pullback_up >= 0) & (pullback_up <= p["trend_pullback_k"])).astype(float)
            - ((slope < 0) & (pullback_dn >= 0) & (pullback_dn <= p["trend_pullback_k"])).astype(float)
        )
        trend_pullback = trend_pullback * np.clip((slope.abs() / 0.01).fillna(0.0), 0.0, 1.0)

        false_break_up = (close.shift(1) > high_n.shift(1)) & (close <= high_n)
        false_break_dn = (close.shift(1) < low_n.shift(1)) & (close >= low_n)
        false_break = false_break_dn.astype(float) - false_break_up.astype(float)
        false_break = false_break.rolling(window=p["false_break_lookback"], min_periods=1).max().fillna(0.0)

        directional = np.sign((breakout_retest + trend_pullback + false_break).to_numpy(dtype=float))
        seq = pd.Series(directional, index=df.index).replace(0, np.nan).ffill().fillna(0.0)
        seq_len = (seq == seq.shift(1)).astype(int).groupby((seq != seq.shift(1)).cumsum()).cumsum() + 1
        seq_weight = np.clip((seq_len / max(1, p["seq_min_len"])).to_numpy(dtype=float), 0.0, 1.0)

        adaptive = np.clip((0.5 + 0.4 * score_trend - 0.2 * score_range - 0.2 * atr_rank).to_numpy(dtype=float), 0.1, 1.0)
        score_raw = (
            0.45 * breakout_retest.to_numpy(dtype=float)
            + 0.40 * trend_pullback.to_numpy(dtype=float)
            + 0.15 * false_break.to_numpy(dtype=float)
        )
        score = score_raw * seq_weight * adaptive
        out = pd.Series(score, index=df.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self.clip_scores(out)

    def propose_entries(self, scores: pd.Series, df: pd.DataFrame, ctx: FamilyContext) -> pd.DataFrame:
        atr_rank = pd.to_numeric(df.get("atr_pct_rank_252", 0.5), errors="coerce").fillna(0.5).astype(float)
        trend = pd.to_numeric(df.get("score_trend", 0.5), errors="coerce").fillna(0.5).astype(float)
        base = float(self.params.get("entry_threshold", 0.30))
        dynamic_thr = np.clip(base + 0.08 * atr_rank - 0.08 * trend, 0.15, 0.65)
        return self.build_entry_frame(
            scores=scores,
            threshold=pd.Series(dynamic_thr, index=scores.index, dtype=float),
            family_name=self.name,
            long_reason="price_long",
            short_reason="price_short",
        )

    def propose_exits(self, position_state: dict[str, Any], df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        return {
            "time_stop_bars": int(self.params.get("time_stop_bars", 24)),
            "stop_atr_multiple": float(self.params.get("stop_atr_multiple", 1.5)),
            "take_profit_atr_multiple": float(self.params.get("take_profit_atr_multiple", 3.0)),
            "trailing_atr_k": float(self.params.get("trailing_atr_k", 1.5)),
        }

    def diagnostics(self, df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        scores = self.compute_scores(df, ctx)
        threshold = float(self.params.get("entry_threshold", 0.30))
        cross = int((scores.abs() >= threshold).sum())
        reversal_horizon = int(self.params.get("false_signal_horizon_bars", 6))
        future_ret = pd.to_numeric(df["close"], errors="coerce").pct_change(reversal_horizon).shift(-reversal_horizon)
        trigger = scores.abs() >= threshold
        wrong_way = ((scores > 0) & (future_ret < 0)) | ((scores < 0) & (future_ret > 0))
        false_rate = float((wrong_way & trigger).sum() / max(1, int(trigger.sum())))
        return {
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=0)),
            "threshold_crossings": int(cross),
            "false_signal_rate_proxy": float(false_rate),
        }

