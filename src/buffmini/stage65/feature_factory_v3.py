"""Stage-65 feature factory v3."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_feature_frame_v3(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    bars = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if bars.empty:
        return pd.DataFrame(), {}
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars.get(col), errors="coerce").astype(float)
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(bars.get("timestamp"), utc=True, errors="coerce")
    out["ret_1"] = bars["close"].pct_change(1)
    out["ret_3"] = bars["close"].pct_change(3)
    out["ret_6"] = bars["close"].pct_change(6)
    out["range_pct"] = (bars["high"] - bars["low"]) / bars["close"].replace(0.0, np.nan)
    out["hlc3"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    out["vol_chg_3"] = bars["volume"].pct_change(3)
    out["vol_z_24"] = _rolling_zscore(bars["volume"], 24)
    out["atr_proxy_14"] = (bars["high"] - bars["low"]).rolling(14, min_periods=3).mean() / bars["close"].replace(0.0, np.nan)
    out["trend_slope_24"] = bars["close"].rolling(24, min_periods=8).mean().pct_change(3)
    out["regime_vol_rank_96"] = out["range_pct"].rolling(96, min_periods=12).rank(pct=True)
    out["liquidity_shock_12"] = out["vol_z_24"] * out["range_pct"].rolling(12, min_periods=4).mean()
    out["crowding_proxy_24"] = out["ret_1"].rolling(24, min_periods=6).mean() / out["ret_1"].rolling(24, min_periods=6).std().replace(0.0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tags = {col: True for col in out.columns if col != "timestamp"}
    return out, tags


def compute_feature_attribution_v3(
    *,
    features: pd.DataFrame,
    label: pd.Series,
) -> pd.DataFrame:
    if features.empty or len(label) != len(features):
        return pd.DataFrame(columns=["feature", "importance", "method"])
    y = pd.to_numeric(label, errors="coerce").fillna(0.0).astype(float)
    rows: list[dict[str, Any]] = []
    for col in [c for c in features.columns if c != "timestamp"]:
        x = pd.to_numeric(features[col], errors="coerce").fillna(0.0).astype(float)
        corr = float(abs(x.corr(y)))
        rows.append({"feature": str(col), "importance": 0.0 if np.isnan(corr) else corr, "method": "abs_corr"})
    out = pd.DataFrame(rows).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    return out


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mean = s.rolling(window, min_periods=max(3, window // 4)).mean()
    std = s.rolling(window, min_periods=max(3, window // 4)).std().replace(0.0, np.nan)
    return (s - mean) / std

