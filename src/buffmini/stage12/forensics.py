"""Stage-12 forensic helpers (execution and metric sanity diagnostics)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def classify_invalid_reason(
    *,
    trade_count: float,
    stability_classification: str,
    usable_windows: int,
    min_usable_windows_valid: int,
) -> str | None:
    """Return deterministic invalid reason label for one combination."""

    if float(trade_count) <= 0.0:
        return "ZERO_TRADE"
    if int(usable_windows) < int(min_usable_windows_valid):
        return "LOW_USABLE_WINDOWS"
    label = str(stability_classification)
    if label in {"INSUFFICIENT_DATA", "INVALID"}:
        return "INSUFFICIENT_DATA"
    return None


def metric_logic_validation(
    *,
    trade_count: float,
    profit_factor: float,
    pnl_values: np.ndarray,
    exp_lcb_reported: float,
    stability_classification: str,
    usable_windows: int,
    min_usable_windows_valid: int,
) -> dict[str, Any]:
    """Validate metric logic invariants without altering semantics."""

    pnl = np.asarray(pnl_values, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    recomputed = _exp_lcb(pnl)
    exp_lcb_ok = bool(abs(float(exp_lcb_reported) - float(recomputed)) <= 1e-12)
    zero_trade_pf_ok = bool(float(trade_count) > 0 or np.isfinite(float(profit_factor)))
    no_losing = bool(pnl.size > 0 and np.all(pnl >= 0.0))
    no_losing_pf_ok = bool((not no_losing) or np.isfinite(float(profit_factor)))
    stability_threshold_ok = not (
        int(usable_windows) < int(min_usable_windows_valid)
        and str(stability_classification) == "STABLE"
    )
    return {
        "exp_lcb_recomputed": float(recomputed),
        "exp_lcb_ok": bool(exp_lcb_ok),
        "zero_trade_pf_ok": bool(zero_trade_pf_ok),
        "no_losing_pf_ok": bool(no_losing_pf_ok),
        "stability_threshold_ok": bool(stability_threshold_ok),
        "all_ok": bool(exp_lcb_ok and zero_trade_pf_ok and no_losing_pf_ok and stability_threshold_ok),
    }


def aggregate_execution_diagnostics(
    matrix: pd.DataFrame,
    suspicious_backtest_ms_threshold: float = 5.0,
) -> dict[str, Any]:
    """Aggregate execution forensic matrix into summary diagnostics."""

    if matrix.empty:
        return {
            "total_combinations": 0,
            "avg_backtest_ms_per_combo": 0.0,
            "min_backtest_ms": 0.0,
            "max_backtest_ms": 0.0,
            "zero_trade_pct": 0.0,
            "invalid_pct": 0.0,
            "invalid_by_reason_pct": {},
            "walkforward_executed_true_pct": 0.0,
            "mc_trigger_rate": 0.0,
            "metric_logic_failures": 0,
            "walkforward_integrity_failures": 0,
            "suspicious_execution": True,
        }

    backtest_ms = pd.to_numeric(matrix["raw_backtest_seconds"], errors="coerce").fillna(0.0) * 1000.0
    total = float(len(matrix))
    invalid_reason_series = matrix["invalid_reason"].astype(str)
    invalid_mask = matrix["invalid_reason"].notna() & (invalid_reason_series != "ZERO_TRADE")
    zero_trade_mask = matrix["invalid_reason"].astype(str) == "ZERO_TRADE"
    walkforward_exec = matrix["walkforward_executed"].astype(bool)
    mc_exec = matrix["MC_executed"].astype(bool)
    logic_ok = matrix["metric_logic_all_ok"].astype(bool)
    wf_integrity_ok = matrix["walkforward_integrity_ok"].astype(bool)

    reason_counts = (
        matrix.loc[invalid_mask, "invalid_reason"]
        .astype(str)
        .value_counts(normalize=True)
        .sort_index()
    )
    invalid_by_reason_pct = {str(k): float(v * 100.0) for k, v in reason_counts.items()}

    avg_backtest_ms = float(backtest_ms.mean())
    summary = {
        "total_combinations": int(len(matrix)),
        "avg_backtest_ms_per_combo": avg_backtest_ms,
        "min_backtest_ms": float(backtest_ms.min()),
        "max_backtest_ms": float(backtest_ms.max()),
        "zero_trade_pct": float(zero_trade_mask.mean() * 100.0),
        "invalid_pct": float(invalid_mask.mean() * 100.0),
        "invalid_by_reason_pct": invalid_by_reason_pct,
        "walkforward_executed_true_pct": float(walkforward_exec.mean() * 100.0),
        "mc_trigger_rate": float(mc_exec.mean() * 100.0),
        "metric_logic_failures": int((~logic_ok).sum()),
        "walkforward_integrity_failures": int((~wf_integrity_ok).sum()),
        "suspicious_execution": bool(avg_backtest_ms < float(suspicious_backtest_ms_threshold)),
    }
    return summary


def classify_stage12_1(
    *,
    diagnostics: dict[str, Any],
    leaderboard: pd.DataFrame,
) -> str:
    """Classify Stage-12.1 forensic outcome."""

    if bool(diagnostics.get("suspicious_execution", False)) or int(diagnostics.get("walkforward_integrity_failures", 0)) > 0:
        return "ENGINE_BUG"
    if int(diagnostics.get("metric_logic_failures", 0)) > 0:
        return "METRIC_BUG"
    valid = leaderboard.loc[leaderboard["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return "TRUE_NO_EDGE"
    stable = valid.loc[valid["stability_classification"] == "STABLE"]
    if stable.empty and float(valid["exp_lcb"].max()) > 0:
        return "EDGE_PRESENT_BUT_INVALIDATED"
    return "TRUE_NO_EDGE"


def write_stage12_1_docs(
    *,
    report_md: Path,
    report_json: Path,
    run_id: str,
    diagnostics: dict[str, Any],
    classification: str,
) -> None:
    """Write Stage-12.1 execution forensic docs."""

    payload = dict(diagnostics)
    payload["run_id"] = str(run_id)
    payload["final_stage12_1_classification"] = str(classification)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-12.1 Execution Forensics & Sanity Audit")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- total_combinations: `{int(payload['total_combinations'])}`")
    lines.append(f"- avg_backtest_ms_per_combo: `{float(payload['avg_backtest_ms_per_combo']):.6f}`")
    lines.append(f"- min_backtest_ms: `{float(payload['min_backtest_ms']):.6f}`")
    lines.append(f"- max_backtest_ms: `{float(payload['max_backtest_ms']):.6f}`")
    lines.append(f"- zero_trade_pct: `{float(payload['zero_trade_pct']):.6f}`")
    lines.append(f"- invalid_pct: `{float(payload['invalid_pct']):.6f}`")
    lines.append(f"- walkforward_executed_true_pct: `{float(payload['walkforward_executed_true_pct']):.6f}`")
    lines.append(f"- mc_trigger_rate: `{float(payload['mc_trigger_rate']):.6f}`")
    lines.append(f"- suspicious_execution: `{bool(payload['suspicious_execution'])}`")
    lines.append(f"- metric_logic_failures: `{int(payload['metric_logic_failures'])}`")
    lines.append(f"- walkforward_integrity_failures: `{int(payload['walkforward_integrity_failures'])}`")
    lines.append("")
    lines.append("## Invalid Reason Distribution")
    lines.append("| reason | pct |")
    lines.append("| --- | ---: |")
    for reason, pct in sorted(payload.get("invalid_by_reason_pct", {}).items()):
        lines.append(f"| {reason} | {float(pct):.6f} |")
    lines.append("")
    lines.append("## Classification")
    lines.append(f"- `{classification}`")
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _exp_lcb(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean
    std = float(np.std(values, ddof=0))
    return float(mean - std / math.sqrt(float(values.size)))


def extract_trade_context_rows(
    *,
    combo_key: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    strategy_key: str,
    strategy_source: str,
    exit_type: str,
    cost_level: str,
    frame: pd.DataFrame,
    trades: pd.DataFrame,
    context_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract per-trade context diagnostics for Stage-12.2."""

    if trades.empty or frame.empty:
        return []
    work = frame.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    high = pd.to_numeric(work["high"], errors="coerce").astype(float)
    low = pd.to_numeric(work["low"], errors="coerce").astype(float)
    close = pd.to_numeric(work["close"], errors="coerce").astype(float)
    atr = pd.to_numeric(work.get("atr_14", 0.0), errors="coerce").astype(float)
    vol_pct = pd.to_numeric(work.get("atr_pct_rank_252", 0.0), errors="coerce").astype(float)
    trend_strength = pd.to_numeric(work.get("trend_strength_stage10", 0.0), errors="coerce").astype(float)
    regime = work.get("regime_label_stage10", pd.Series(["UNKNOWN"] * len(work))).astype(str)
    score_trend = pd.to_numeric(work.get("score_trend", 0.0), errors="coerce").astype(float)
    score_range = pd.to_numeric(work.get("score_range", 0.0), errors="coerce").astype(float)
    score_vol = pd.to_numeric(work.get("score_vol_expansion", 0.0), errors="coerce").astype(float)
    score_chop = pd.to_numeric(work.get("score_chop", 0.0), errors="coerce").astype(float)

    rows: list[dict[str, Any]] = []
    family = _strategy_family(strategy_key=strategy_key, strategy_name=strategy, strategy_source=strategy_source)
    for trade in trades.to_dict(orient="records"):
        entry_time = _to_utc(trade.get("entry_time"))
        exit_time = _to_utc(trade.get("exit_time"))
        if entry_time is None or exit_time is None:
            continue
        entry_idx = int(ts.searchsorted(entry_time, side="left"))
        if entry_idx >= len(work):
            entry_idx = len(work) - 1
        exit_idx = int(ts.searchsorted(exit_time, side="right") - 1)
        exit_idx = max(entry_idx, min(exit_idx, len(work) - 1))
        if entry_idx < 0 or exit_idx < 0:
            continue
        entry_price = float(trade.get("entry_price", np.nan))
        side = str(trade.get("side", "long"))
        if not np.isfinite(entry_price) or entry_price == 0.0:
            continue
        path_high = float(np.nanmax(high.iloc[entry_idx : exit_idx + 1]))
        path_low = float(np.nanmin(low.iloc[entry_idx : exit_idx + 1]))
        if side == "long":
            mae = max(0.0, (entry_price - path_low) / entry_price)
            mfe = max(0.0, (path_high - entry_price) / entry_price)
        else:
            mae = max(0.0, (path_high - entry_price) / entry_price)
            mfe = max(0.0, (entry_price - path_low) / entry_price)

        vol_value = _safe_num(vol_pct.iloc[entry_idx], default=0.0)
        trend_value = _safe_num(trend_strength.iloc[entry_idx], default=0.0)
        atr_ratio = _safe_num(atr.iloc[entry_idx] / close.iloc[entry_idx], default=0.0)
        regime_align = _regime_alignment(
            family=family,
            trend=_safe_num(score_trend.iloc[entry_idx], default=0.0),
            range_score=_safe_num(score_range.iloc[entry_idx], default=0.0),
            vol_score=_safe_num(score_vol.iloc[entry_idx], default=0.0),
        )
        chop_value = _safe_num(score_chop.iloc[entry_idx], default=0.0)
        context_score = _context_score(
            family=family,
            regime_alignment=regime_align,
            volatility_percentile=vol_value,
            trend_strength=trend_value,
            chop_score=chop_value,
            context_cfg=context_cfg,
        )
        pnl = _safe_num(trade.get("pnl", 0.0), default=0.0)
        rows.append(
            {
                "combo_key": combo_key,
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "strategy_key": strategy_key,
                "strategy_source": strategy_source,
                "exit_type": exit_type,
                "cost_level": cost_level,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "pnl": pnl,
                "is_winner": bool(pnl > 0.0),
                "regime": str(regime.iloc[entry_idx]),
                "volatility_percentile": vol_value,
                "ATR_ratio": atr_ratio,
                "trend_strength": trend_value,
                "holding_duration": int(max(0, exit_idx - entry_idx)),
                "MAE": mae,
                "MFE": mfe,
                "time_of_day": int(entry_time.hour),
                "regime_alignment": regime_align,
                "chop_score": chop_value,
                "context_score": context_score,
            }
        )
    return rows


def summarize_signal_forensics(
    *,
    trade_context: pd.DataFrame,
    context_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build Stage-12.2 winner/loser diagnostics and verdict."""

    if trade_context.empty:
        empty = pd.DataFrame()
        return empty, {
            "total_trades": 0,
            "losing_trades": 0,
            "winning_trades": 0,
            "avg_context_score_winners": 0.0,
            "avg_context_score_losers": 0.0,
            "context_score_diff": 0.0,
            "context_separation_effect_size": 0.0,
            "context_separation_detected": False,
            "final_stage12_2_verdict": "RANDOM_NOISE",
        }, empty

    df = trade_context.copy()
    df["is_winner"] = df["is_winner"].astype(bool)
    winners = df.loc[df["is_winner"]]
    losers = df.loc[~df["is_winner"]]
    false_positive_map = losers.copy()

    per_strategy_rows: list[dict[str, Any]] = []
    for strategy, group in df.groupby("strategy", sort=True):
        win = group.loc[group["is_winner"]]
        loss = group.loc[~group["is_winner"]]
        row = {
            "strategy": str(strategy),
            "samples_total": int(len(group)),
            "samples_win": int(len(win)),
            "samples_loss": int(len(loss)),
            "mean_regime_alignment_win": _safe_mean(win.get("regime_alignment", pd.Series(dtype=float))),
            "mean_regime_alignment_loss": _safe_mean(loss.get("regime_alignment", pd.Series(dtype=float))),
            "mean_volatility_win": _safe_mean(win.get("volatility_percentile", pd.Series(dtype=float))),
            "mean_volatility_loss": _safe_mean(loss.get("volatility_percentile", pd.Series(dtype=float))),
            "mean_ATR_ratio_win": _safe_mean(win.get("ATR_ratio", pd.Series(dtype=float))),
            "mean_ATR_ratio_loss": _safe_mean(loss.get("ATR_ratio", pd.Series(dtype=float))),
            "mean_holding_duration_win": _safe_mean(win.get("holding_duration", pd.Series(dtype=float))),
            "mean_holding_duration_loss": _safe_mean(loss.get("holding_duration", pd.Series(dtype=float))),
            "mean_MFE_win": _safe_mean(win.get("MFE", pd.Series(dtype=float))),
            "mean_MFE_loss": _safe_mean(loss.get("MFE", pd.Series(dtype=float))),
            "mean_MAE_win": _safe_mean(win.get("MAE", pd.Series(dtype=float))),
            "mean_MAE_loss": _safe_mean(loss.get("MAE", pd.Series(dtype=float))),
            "mean_context_score_win": _safe_mean(win.get("context_score", pd.Series(dtype=float))),
            "mean_context_score_loss": _safe_mean(loss.get("context_score", pd.Series(dtype=float))),
            "separation_score": _cohen_d(
                win.get("context_score", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
                loss.get("context_score", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
            ),
        }
        per_strategy_rows.append(row)
    by_strategy = pd.DataFrame(per_strategy_rows).sort_values("strategy").reset_index(drop=True)

    win_scores = winners.get("context_score", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False)
    loss_scores = losers.get("context_score", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False)
    avg_win = _safe_mean(pd.Series(win_scores, dtype=float))
    avg_loss = _safe_mean(pd.Series(loss_scores, dtype=float))
    effect = _cohen_d(win_scores, loss_scores)
    min_samples = int(context_cfg.get("min_samples", 30))
    effect_threshold = float(context_cfg.get("separation_effect_size_threshold", 0.10))
    context_separation_detected = bool(
        len(win_scores) >= min_samples
        and len(loss_scores) >= min_samples
        and avg_win > avg_loss
        and effect >= effect_threshold
    )
    verdict = _stage12_2_verdict(
        total=int(len(df)),
        winners=int(len(winners)),
        losers=int(len(losers)),
        separation_detected=context_separation_detected,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
    summary = {
        "total_trades": int(len(df)),
        "losing_trades": int(len(losers)),
        "winning_trades": int(len(winners)),
        "avg_context_score_winners": avg_win,
        "avg_context_score_losers": avg_loss,
        "context_score_diff": float(avg_win - avg_loss),
        "context_separation_effect_size": effect,
        "context_separation_detected": context_separation_detected,
        "final_stage12_2_verdict": verdict,
    }
    return false_positive_map, summary, by_strategy


def write_stage12_2_docs(
    *,
    report_md: Path,
    report_json: Path,
    run_id: str,
    summary: dict[str, Any],
    by_strategy: pd.DataFrame,
) -> None:
    """Write Stage-12.2 signal forensic docs."""

    payload = dict(summary)
    payload["run_id"] = str(run_id)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-12.2 Signal Forensics & Context Modeling Foundation")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- total_trades: `{int(payload['total_trades'])}`")
    lines.append(f"- winning_trades: `{int(payload['winning_trades'])}`")
    lines.append(f"- losing_trades: `{int(payload['losing_trades'])}`")
    lines.append(f"- avg_context_score_winners: `{float(payload['avg_context_score_winners']):.6f}`")
    lines.append(f"- avg_context_score_losers: `{float(payload['avg_context_score_losers']):.6f}`")
    lines.append(f"- context_score_diff: `{float(payload['context_score_diff']):.6f}`")
    lines.append(f"- context_separation_effect_size: `{float(payload['context_separation_effect_size']):.6f}`")
    lines.append(f"- context_separation_detected: `{bool(payload['context_separation_detected'])}`")
    lines.append(f"- final_stage12_2_verdict: `{payload['final_stage12_2_verdict']}`")
    lines.append("")
    lines.append("## Winner vs Loser Separation by Strategy")
    if by_strategy.empty:
        lines.append("- no trades available")
    else:
        lines.append("| strategy | samples_win | samples_loss | mean_context_win | mean_context_loss | separation_score |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in by_strategy.to_dict(orient="records"):
            lines.append(
                f"| {row['strategy']} | {int(row['samples_win'])} | {int(row['samples_loss'])} | "
                f"{float(row['mean_context_score_win']):.6f} | {float(row['mean_context_score_loss']):.6f} | "
                f"{float(row['separation_score']):.6f} |"
            )
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _strategy_family(strategy_key: str, strategy_name: str, strategy_source: str) -> str:
    name = str(strategy_key)
    if str(strategy_source) == "stage10":
        if name in {"MA_SlopePullback"}:
            return "trend"
        if name in {"BreakoutRetest", "VolCompressionBreakout"}:
            return "breakout"
        return "mean_reversion"
    text = str(strategy_name).lower()
    if "mean reversion" in text:
        return "mean_reversion"
    if "breakout" in text:
        return "breakout"
    return "trend"


def _regime_alignment(*, family: str, trend: float, range_score: float, vol_score: float) -> float:
    if family == "trend":
        return float(np.clip(trend, 0.0, 1.0))
    if family == "mean_reversion":
        return float(np.clip(range_score, 0.0, 1.0))
    return float(np.clip(vol_score, 0.0, 1.0))


def _context_score(
    *,
    family: str,
    regime_alignment: float,
    volatility_percentile: float,
    trend_strength: float,
    chop_score: float,
    context_cfg: dict[str, Any],
) -> float:
    wr = float(context_cfg.get("regime_alignment_weight", 0.40))
    wv = float(context_cfg.get("volatility_alignment_weight", 0.25))
    wt = float(context_cfg.get("trend_strength_weight", 0.25))
    wc = float(context_cfg.get("chop_penalty_weight", 0.20))
    target_vol = 0.65 if family in {"trend", "breakout"} else 0.35
    vol_align = float(np.clip(1.0 - (abs(float(volatility_percentile) - target_vol) / 0.5), 0.0, 1.0))
    trend_norm = float(np.clip(float(trend_strength) / 0.02, 0.0, 1.0))
    raw = (wr * float(regime_alignment)) + (wv * vol_align) + (wt * trend_norm) - (wc * float(chop_score))
    return float(np.clip(raw, 0.0, 1.0))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 2 or right.size < 2:
        return 0.0
    l_mean = float(np.mean(left))
    r_mean = float(np.mean(right))
    l_var = float(np.var(left, ddof=1))
    r_var = float(np.var(right, ddof=1))
    pooled = ((left.size - 1) * l_var + (right.size - 1) * r_var) / max(1.0, float(left.size + right.size - 2))
    if pooled <= 0:
        return 0.0
    return float((l_mean - r_mean) / math.sqrt(pooled))


def _stage12_2_verdict(*, total: int, winners: int, losers: int, separation_detected: bool, avg_win: float, avg_loss: float) -> str:
    if total <= 0:
        return "RANDOM_NOISE"
    if separation_detected:
        return "CONTEXT_DEPENDENT_EDGE"
    if losers > winners and avg_win <= avg_loss:
        if winners <= max(5, int(0.2 * total)):
            return "STRATEGY_DESIGN_FLAW"
        return "STRUCTURAL_FALSE_POSITIVES"
    return "RANDOM_NOISE"


def _safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _to_utc(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)
