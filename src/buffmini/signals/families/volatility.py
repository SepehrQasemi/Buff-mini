"""Stage-13.3 volatility/compression family."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.signals.family_base import FamilyContext, SignalFamily


class VolatilityCompressionFamily(SignalFamily):
    """Volatility transition family with soft regime weighting."""

    name = "volatility"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def required_features(self) -> list[str]:
        return [
            "timestamp",
            "close",
            "atr_14",
            "atr_pct_rank_252",
            "bb_bandwidth_20",
            "bb_bandwidth_z_120",
            "ema_slope_50",
            "score_vol_expansion",
            "score_vol_compression",
        ]

    def compute_scores(self, df: pd.DataFrame, ctx: FamilyContext) -> pd.Series:
        self.validate_frame(df)
        p = {
            "compression_z": float(self.params.get("compression_z", -0.8)),
            "expansion_z": float(self.params.get("expansion_z", 0.8)),
            "exhaustion_rank": float(self.params.get("exhaustion_rank", 0.90)),
            "atr_slope_window": int(self.params.get("atr_slope_window", 12)),
        }
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        atr = pd.to_numeric(df["atr_14"], errors="coerce").replace(0.0, np.nan).astype(float)
        atr_rank = pd.to_numeric(df["atr_pct_rank_252"], errors="coerce").fillna(0.5).astype(float)
        bw_z = pd.to_numeric(df["bb_bandwidth_z_120"], errors="coerce").fillna(0.0).astype(float)
        bw = pd.to_numeric(df["bb_bandwidth_20"], errors="coerce").fillna(0.0).astype(float)
        slope = pd.to_numeric(df["ema_slope_50"], errors="coerce").fillna(0.0).astype(float)
        vol_exp = pd.to_numeric(df["score_vol_expansion"], errors="coerce").fillna(0.0).astype(float)
        vol_cmp = pd.to_numeric(df["score_vol_compression"], errors="coerce").fillna(0.0).astype(float)

        compression = (bw_z < p["compression_z"]).fillna(False)
        expansion = (bw_z > p["expansion_z"]).fillna(False)
        transition = compression.shift(1, fill_value=False) & expansion
        breakout_dir = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))
        contraction_breakout = pd.Series(transition.to_numpy(dtype=float) * breakout_dir, index=df.index, dtype=float)

        exhaustion = atr_rank >= p["exhaustion_rank"]
        exhaustion_revert = pd.Series(
            np.where(exhaustion & (slope > 0), -1.0, np.where(exhaustion & (slope < 0), 1.0, 0.0)),
            index=df.index,
            dtype=float,
        )

        atr_slope = atr.diff(max(1, p["atr_slope_window"])).fillna(0.0)
        atr_slope_mod = np.tanh((atr_slope / (atr + 1e-12)).to_numpy(dtype=float) * 2.0)

        regime_weight = np.clip((0.6 * vol_cmp + 0.4 * vol_exp).to_numpy(dtype=float), 0.1, 1.0)
        score_raw = (
            0.45 * contraction_breakout.to_numpy(dtype=float)
            + 0.35 * exhaustion_revert.to_numpy(dtype=float)
            + 0.20 * atr_slope_mod
        )
        score = score_raw * regime_weight
        return self.clip_scores(pd.Series(score, index=df.index, dtype=float).fillna(0.0))

    def propose_entries(self, scores: pd.Series, df: pd.DataFrame, ctx: FamilyContext) -> pd.DataFrame:
        atr_rank = pd.to_numeric(df.get("atr_pct_rank_252", 0.5), errors="coerce").fillna(0.5).astype(float)
        base = float(self.params.get("entry_threshold", 0.28))
        thr = np.clip(base + 0.1 * (atr_rank - 0.5), 0.12, 0.65)
        return self.build_entry_frame(
            scores=scores,
            threshold=pd.Series(thr, index=scores.index, dtype=float),
            family_name=self.name,
            long_reason="vol_long",
            short_reason="vol_short",
        )

    def propose_exits(self, position_state: dict[str, Any], df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        vol_exp = pd.to_numeric(df.get("score_vol_expansion", 0.0), errors="coerce").fillna(0.0)
        med_exp = float(vol_exp.median()) if len(vol_exp) else 0.0
        base_stop = float(self.params.get("stop_atr_multiple", 1.5))
        widen = float(self.params.get("vol_wide_stop_mult", 1.10))
        stop_mult = base_stop * (widen if med_exp > 0.6 else 1.0)
        return {
            "time_stop_bars": int(self.params.get("time_stop_bars", 24)),
            "stop_atr_multiple": float(stop_mult),
            "take_profit_atr_multiple": float(self.params.get("take_profit_atr_multiple", 3.0)),
            "trailing_atr_k": float(self.params.get("trailing_atr_k", 1.5)),
        }

    def diagnostics(self, df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        scores = self.compute_scores(df, ctx)
        threshold = float(self.params.get("entry_threshold", 0.28))
        triggered = scores.abs() >= threshold
        return {
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=0)),
            "threshold_crossings": int(triggered.sum()),
            "compression_share": float((pd.to_numeric(df["score_vol_compression"], errors="coerce").fillna(0.0) > 0.5).mean()),
        }

