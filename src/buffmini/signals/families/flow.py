"""Stage-13.4 flow/liquidity family with masked overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.signals.family_base import FamilyContext, SignalFamily


class FlowLiquidityFamily(SignalFamily):
    """Flow family using free-data proxies and optional futures overlays."""

    name = "flow"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def required_features(self) -> list[str]:
        return [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr_14",
        ]

    def compute_scores(self, df: pd.DataFrame, ctx: FamilyContext) -> pd.Series:
        self.validate_frame(df)
        p = {
            "volume_window": int(self.params.get("volume_window", 48)),
            "anomaly_cap": float(self.params.get("anomaly_cap", 3.0)),
            "riskoff_hard_gate": bool(self.params.get("riskoff_hard_gate", False)),
        }
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        open_ = pd.to_numeric(df["open"], errors="coerce").astype(float)
        high = pd.to_numeric(df["high"], errors="coerce").astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").astype(float)
        volume = pd.to_numeric(df["volume"], errors="coerce").astype(float)
        atr = pd.to_numeric(df["atr_14"], errors="coerce").replace(0.0, np.nan).astype(float)

        vol_med = volume.rolling(window=p["volume_window"], min_periods=max(5, p["volume_window"] // 4)).median()
        vol_ratio = (volume / (vol_med + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_z = ((vol_ratio - vol_ratio.rolling(48, min_periods=10).mean()) / (vol_ratio.rolling(48, min_periods=10).std(ddof=0) + 1e-12)).fillna(0.0)
        volume_anomaly = np.clip(vol_z.to_numpy(dtype=float) / p["anomaly_cap"], -1.0, 1.0)

        bar_range = (high - low).abs()
        effort_vs_result = ((vol_ratio - 1.0) - (bar_range / (atr + 1e-12) - 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        effort_score = np.clip(effort_vs_result.to_numpy(dtype=float) / 4.0, -1.0, 1.0)
        trend_dir = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))

        # Funding/OI overlays are optional and may be NaN outside recent windows.
        funding_pos = self._num_series(df, "funding_extreme_pos", default=0.0).to_numpy(dtype=float) > 0
        funding_neg = self._num_series(df, "funding_extreme_neg", default=0.0).to_numpy(dtype=float) > 0
        crowd_long = self._num_series(df, "crowd_long_risk", default=0.0).to_numpy(dtype=float) > 0
        crowd_short = self._num_series(df, "crowd_short_risk", default=0.0).to_numpy(dtype=float) > 0
        oi_active = self._num_series(df, "oi_active", default=0.0).to_numpy(dtype=float) > 0

        overlay = np.zeros(len(df), dtype=float)
        overlay = np.where(funding_pos, overlay - 0.25, overlay)
        overlay = np.where(funding_neg, overlay + 0.25, overlay)
        overlay = np.where(crowd_long & oi_active, overlay - 0.35, overlay)
        overlay = np.where(crowd_short & oi_active, overlay + 0.35, overlay)

        base_score = 0.45 * (volume_anomaly * trend_dir) + 0.35 * effort_score + 0.20 * overlay
        out = pd.Series(np.clip(base_score, -1.0, 1.0), index=df.index, dtype=float).fillna(0.0)
        if p["riskoff_hard_gate"]:
            # Off by default: optional explicit hard gate if user enables it.
            riskoff = (crowd_long | crowd_short) & oi_active
            out = out.mask(riskoff, 0.0)
        return self.clip_scores(out)

    def propose_entries(self, scores: pd.Series, df: pd.DataFrame, ctx: FamilyContext) -> pd.DataFrame:
        threshold = float(self.params.get("entry_threshold", 0.30))
        return self.build_entry_frame(
            scores=scores,
            threshold=threshold,
            family_name=self.name,
            long_reason="flow_long",
            short_reason="flow_short",
        )

    def propose_exits(self, position_state: dict[str, Any], df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        return {
            "time_stop_bars": int(self.params.get("time_stop_bars", 24)),
            "stop_atr_multiple": float(self.params.get("stop_atr_multiple", 1.5)),
            "take_profit_atr_multiple": float(self.params.get("take_profit_atr_multiple", 3.0)),
            "trailing_atr_k": float(self.params.get("trailing_atr_k", 1.5)),
            "risk_off_penalty": float(self.params.get("risk_off_penalty", 0.85)),
        }

    def diagnostics(self, df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        scores = self.compute_scores(df, ctx)
        return {
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=0)),
            "threshold_crossings": int((scores.abs() >= float(self.params.get("entry_threshold", 0.30))).sum()),
            "oi_active_share": float(self._num_series(df, "oi_active", default=0.0).mean()),
        }

    @staticmethod
    def _num_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(float(default))
        return pd.Series(float(default), index=df.index, dtype=float)
