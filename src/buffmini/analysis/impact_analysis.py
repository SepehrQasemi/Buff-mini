"""Stage-9 impact analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.baselines.stage0 import (
    bollinger_mean_reversion,
    donchian_breakout,
    generate_signals,
    range_breakout_with_ema_trend_filter,
    rsi_mean_reversion,
    trend_pullback,
)
from buffmini.discovery.dsl import select_signals_by_regime


CONDITIONS: tuple[str, ...] = (
    "funding_extreme_pos",
    "funding_extreme_neg",
    "crowd_long_risk",
    "crowd_short_risk",
)


def compute_forward_returns(frame: pd.DataFrame, horizons: tuple[int, ...] = (24, 72)) -> pd.DataFrame:
    """Append forward return columns for specified hourly horizons."""

    data = frame.copy()
    close = pd.to_numeric(data["close"], errors="coerce").astype(float)
    for horizon in horizons:
        h = int(horizon)
        data[f"forward_return_{h}h"] = close.shift(-h) / close - 1.0
    return data


def conditional_stats(frame: pd.DataFrame, condition_col: str, value_col: str) -> dict[str, float]:
    """Compute distribution stats for one conditional subset."""

    condition = pd.to_numeric(frame[condition_col], errors="coerce").fillna(0) > 0
    values = pd.to_numeric(frame[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    subset = values.loc[condition].dropna()

    if subset.empty:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}

    return {
        "count": float(len(subset)),
        "mean": float(subset.mean()),
        "median": float(subset.median()),
        "p05": float(subset.quantile(0.05)),
        "p95": float(subset.quantile(0.95)),
    }


def bootstrap_median_difference(
    frame: pd.DataFrame,
    condition_col: str,
    value_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for median difference: median(condition) - median(not condition)."""

    condition = pd.to_numeric(frame[condition_col], errors="coerce").fillna(0) > 0
    values = pd.to_numeric(frame[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    a = values.loc[condition].dropna().to_numpy(dtype=float)
    b = values.loc[~condition].dropna().to_numpy(dtype=float)
    if a.size == 0 or b.size == 0:
        return {"median_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        samp_a = a[rng.integers(0, a.size, size=a.size)]
        samp_b = b[rng.integers(0, b.size, size=b.size)]
        diffs[idx] = float(np.median(samp_a) - np.median(samp_b))

    return {
        "median_diff": float(np.median(a) - np.median(b)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def analyze_symbol_impact(
    features: pd.DataFrame,
    symbol: str,
    seed: int = 42,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Run Stage-9 impact analysis for one symbol frame."""

    prepared = compute_forward_returns(features)
    rows: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        if cond not in prepared.columns:
            continue
        for horizon in (24, 72):
            ret_col = f"forward_return_{horizon}h"
            cond_stats = conditional_stats(prepared, cond, ret_col)
            inv_cond = (pd.to_numeric(prepared[cond], errors="coerce").fillna(0) <= 0).astype(int)
            base = prepared.copy()
            base[f"not_{cond}"] = inv_cond
            base_stats = conditional_stats(base, f"not_{cond}", ret_col)
            boot = bootstrap_median_difference(prepared, cond, ret_col, n_boot=n_boot, seed=seed + horizon)
            rows.append(
                {
                    "symbol": symbol,
                    "condition": cond,
                    "horizon": f"{horizon}h",
                    "count_condition": int(cond_stats["count"]),
                    "count_base": int(base_stats["count"]),
                    "median_condition": float(cond_stats["median"]),
                    "median_base": float(base_stats["median"]),
                    "median_diff": float(boot["median_diff"]),
                    "ci_low": float(boot["ci_low"]),
                    "ci_high": float(boot["ci_high"]),
                    "mean_condition": float(cond_stats["mean"]),
                    "mean_base": float(base_stats["mean"]),
                    "p05_condition": float(cond_stats["p05"]),
                    "p95_condition": float(cond_stats["p95"]),
                }
            )

    corr_24 = prepared[["funding_z_30", "oi_z_30", "forward_return_24h"]].copy()
    corr_24 = corr_24.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr_matrix = corr_24.corr(numeric_only=True)
    corr_funding = (
        float(corr_matrix.loc["funding_z_30", "forward_return_24h"])
        if "funding_z_30" in corr_matrix.index and "forward_return_24h" in corr_matrix.columns
        else 0.0
    )
    corr_oi = (
        float(corr_matrix.loc["oi_z_30", "forward_return_24h"])
        if "oi_z_30" in corr_matrix.index and "forward_return_24h" in corr_matrix.columns
        else 0.0
    )

    if rows:
        strongest = sorted(rows, key=lambda item: abs(float(item["median_diff"])), reverse=True)[0]
    else:
        strongest = {
            "symbol": symbol,
            "condition": "none",
            "horizon": "24h",
            "median_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    return {
        "symbol": symbol,
        "rows": rows,
        "corr_funding_z30_vs_fwd24": corr_funding,
        "corr_oi_z30_vs_fwd24": corr_oi,
        "best_effect": strongest,
    }


def summarize_data_quality(
    symbol: str,
    funding_meta: dict[str, Any],
    oi_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build concise data-quality summary row for report JSON."""

    return {
        "symbol": symbol,
        "funding": {
            "start_ts": funding_meta.get("start_ts"),
            "end_ts": funding_meta.get("end_ts"),
            "row_count": int(funding_meta.get("row_count", 0) or 0),
            "gaps_count": int(funding_meta.get("gaps_count", 0) or 0),
        },
        "open_interest": {
            "start_ts": oi_meta.get("start_ts"),
            "end_ts": oi_meta.get("end_ts"),
            "row_count": int(oi_meta.get("row_count", 0) or 0),
            "gaps_count": int(oi_meta.get("gaps_count", 0) or 0),
        },
    }


def compute_dsl_trade_count_ratio(frame: pd.DataFrame) -> dict[str, float]:
    """Compare baseline vs DSL-lite entry counts on one feature frame."""

    strategy_specs = [
        donchian_breakout(),
        rsi_mean_reversion(),
        trend_pullback(),
        bollinger_mean_reversion(),
        range_breakout_with_ema_trend_filter(),
    ]
    pair_by_name = {
        "Donchian Breakout": rsi_mean_reversion(),
        "RSI Mean Reversion": donchian_breakout(),
        "Bollinger Mean Reversion": range_breakout_with_ema_trend_filter(),
        "Range Breakout w/ EMA Trend Filter": bollinger_mean_reversion(),
    }

    baseline_entries = 0
    dsl_entries = 0
    for spec in strategy_specs:
        primary = generate_signals(frame, spec, gating_mode="none")
        selected = primary
        paired = pair_by_name.get(spec.name)
        if paired is not None:
            alternate = generate_signals(frame, paired, gating_mode="none")
            selected = select_signals_by_regime(
                frame=frame,
                primary_signal=primary,
                alternate_signal=alternate,
                use_funding_selector=True,
                use_oi_selector=True,
            )

        baseline_entries += int(_count_entries(primary))
        dsl_entries += int(_count_entries(selected))

    ratio = float(dsl_entries / baseline_entries) if baseline_entries > 0 else 1.0
    return {
        "baseline_entries": float(baseline_entries),
        "dsl_entries": float(dsl_entries),
        "ratio": float(ratio),
    }


def _count_entries(signal: pd.Series) -> int:
    s = pd.Series(signal).fillna(0).astype(int)
    entered = s.ne(0) & s.shift(1).fillna(0).eq(0)
    return int(entered.sum())
