"""Real executable validation helpers for Stage-52 candidates."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import get_universe_end
from buffmini.constants import CANONICAL_DATA_DIR, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.continuity import continuity_report
from buffmini.data.derived_tf import get_timeframe
from buffmini.data.store import build_data_store
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage45 import (
    compute_liquidity_map,
    compute_market_structure_engine,
    compute_volatility_regime_engine,
)
from buffmini.stage46 import (
    compute_crowding_layer,
    compute_flow_regime_engine,
    compute_mtf_bias_completion,
    compute_trade_geometry_layer,
)
from buffmini.stage52 import default_geometry_for_family
from buffmini.utils.hashing import stable_hash
from buffmini.validation.walkforward_v2 import aggregate_windows, build_windows


def resolve_validation_candidate(
    *,
    runs_dir: Path,
    stage28_run_id: str,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve one deterministic validation candidate from Stage-52 artifacts."""

    if not str(stage28_run_id).strip():
        return {}
    stage52_path = Path(runs_dir) / str(stage28_run_id) / "stage52" / "setup_candidates_v2.csv"
    if not stage52_path.exists():
        return {}
    frame = pd.read_csv(stage52_path)
    if frame.empty:
        return {}

    preferred_id = ""
    if docs_dir is not None:
        stage53_summary = _load_json(Path(docs_dir) / "stage53_summary.json")
        preferred_id = str(stage53_summary.get("validated_candidate_id", "")).strip()
        if not preferred_id:
            stage52_summary = _load_json(Path(docs_dir) / "stage52_summary.json")
            preferred_id = str(stage52_summary.get("representative_candidate_id", "")).strip()

    frame["beam_score"] = pd.to_numeric(frame.get("beam_score", 0.0), errors="coerce").fillna(0.0)
    frame["cost_edge_proxy"] = pd.to_numeric(frame.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0)
    eligible_source = frame["eligible_for_replay"] if "eligible_for_replay" in frame.columns else pd.Series([True] * len(frame), index=frame.index)
    frame["eligible_for_replay"] = eligible_source.fillna(True).astype(bool)
    if preferred_id:
        preferred = frame.loc[frame.get("candidate_id", "").astype(str) == preferred_id].copy()
        if not preferred.empty:
            return hydrate_candidate_record(dict(preferred.iloc[0].to_dict()))

    ranked = frame.sort_values(
        ["eligible_for_replay", "beam_score", "cost_edge_proxy", "candidate_id"],
        ascending=[False, False, False, True],
    )
    return hydrate_candidate_record(dict(ranked.iloc[0].to_dict())) if not ranked.empty else {}


def hydrate_candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one Stage-52 candidate row into an executable record."""

    row = dict(record or {})
    family = str(row.get("family", "")).strip()
    timeframe = str(row.get("timeframe", "1h")).strip() or "1h"
    geometry = _safe_mapping(row.get("geometry"))
    if not geometry and family:
        try:
            geometry = default_geometry_for_family(family=family, timeframe=timeframe)
        except Exception:
            geometry = _generic_runtime_geometry(family=family, timeframe=timeframe)
    rr_model = _safe_mapping(row.get("rr_model"))
    if not rr_model and geometry:
        stop_distance_pct = _safe_float(geometry.get("stop_distance_pct", 0.0))
        first_target_pct = _safe_float(geometry.get("first_target_pct", 0.0))
        rr_model = {
            "first_target_rr": float(first_target_pct / max(stop_distance_pct, 1e-9)),
        }
    row["family"] = family
    row["timeframe"] = timeframe
    row["geometry"] = geometry
    row["rr_model"] = rr_model
    row["candidate_id"] = str(row.get("candidate_id", "")).strip()
    row["entry_logic"] = str(row.get("entry_logic", geometry.get("entry_logic", ""))).strip()
    row["stop_logic"] = str(row.get("stop_logic", geometry.get("stop_logic", ""))).strip()
    row["target_logic"] = str(row.get("target_logic", geometry.get("target_logic", ""))).strip()
    row["hold_logic"] = str(row.get("hold_logic", geometry.get("hold_logic", ""))).strip()
    return row


def load_candidate_market_frame(
    config: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one runtime frame with continuity diagnostics and explicit fallback semantics."""

    universe = dict(config.get("universe", {}))
    data_cfg = dict(config.get("data", {}))
    continuity_cfg = dict(data_cfg.get("continuity", {}))
    start = universe.get("start")
    end = get_universe_end(config)
    resolved_end_ts = universe.get("resolved_end_ts") or end or ""

    frame = pd.DataFrame()
    run_cfg = dict(config.get("research_run", {}) or {})
    data_source = str(run_cfg.get("data_source", "live")).strip().lower() or "live"
    load_mode = "configured_store"
    load_errors: list[str] = []
    if data_source == "canonical":
        try:
            canonical = get_timeframe(
                symbol=symbol,
                timeframe=timeframe,
                canonical_dir=CANONICAL_DATA_DIR,
                derived_dir=DERIVED_DATA_DIR,
            )
            frame = canonical.frame.copy()
            load_mode = "canonical_snapshot"
        except Exception as exc:
            load_errors.append(f"canonical_snapshot:{exc}")
    else:
        try:
            store = build_data_store(
                backend=str(data_cfg.get("backend", "parquet")),
                data_dir=RAW_DATA_DIR,
                base_timeframe=str(universe.get("base_timeframe", "1h")),
                resample_source=str(data_cfg.get("resample_source", "direct")),
                derived_dir=DERIVED_DATA_DIR,
                partial_last_bucket=bool(data_cfg.get("partial_last_bucket", False)),
                resolved_end_ts=str(resolved_end_ts),
            )
            frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
        except Exception as exc:
            load_errors.append(f"configured_store:{exc}")

        if frame.empty and str(timeframe) != "1m":
            try:
                fallback_store = build_data_store(
                    backend=str(data_cfg.get("backend", "parquet")),
                    data_dir=RAW_DATA_DIR,
                    base_timeframe="1m",
                    resample_source="base",
                    derived_dir=DERIVED_DATA_DIR,
                    partial_last_bucket=bool(data_cfg.get("partial_last_bucket", False)),
                    resolved_end_ts=str(resolved_end_ts),
                )
                frame = fallback_store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
                load_mode = "fallback_resampled_from_1m"
            except Exception as exc:
                load_errors.append(f"fallback_resampled_from_1m:{exc}")

    if not frame.empty and "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        mask = pd.Series(True, index=frame.index, dtype=bool)
        if str(start or "").strip():
            mask &= ts >= pd.to_datetime(start, utc=True, errors="coerce")
        if str(end or "").strip():
            mask &= ts <= pd.to_datetime(end, utc=True, errors="coerce")
        frame = frame.loc[mask].copy().reset_index(drop=True)

    continuity = continuity_report(
        frame,
        timeframe=str(timeframe),
        max_gap_bars=int(max(0, continuity_cfg.get("max_gap_bars", 0))),
    )
    frozen_mode = bool(config.get("reproducibility", {}).get("frozen_research_mode", False))
    require_resolved_end_ts = bool(config.get("reproducibility", {}).get("require_resolved_end_ts", False))
    strict_mode = bool(continuity_cfg.get("strict_mode", False)) or frozen_mode
    fail_on_gap = bool(continuity_cfg.get("fail_on_gap", False))
    resolved_end_required_effective = bool(require_resolved_end_ts or frozen_mode)
    resolved_end_missing = bool(resolved_end_required_effective and not str(resolved_end_ts).strip())
    continuity_blocked = bool((fail_on_gap or strict_mode) and not continuity.get("passes_strict", True))
    runtime_truth_reason = "MISSING_RESOLVED_END_TS" if resolved_end_missing else ""
    runtime_truth_blocked = bool(runtime_truth_reason)
    start_ts = ""
    end_ts = ""
    if not frame.empty and "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            start_ts = str(ts.min().isoformat())
            end_ts = str(ts.max().isoformat())
    continuity_reason = ""
    if continuity_blocked:
        continuity_reason = (
            f"gap_count={int(continuity.get('gap_count', 0))},"
            f"largest_gap_bars={int(continuity.get('largest_gap_bars', 0))},"
            f"max_gap_bars={int(continuity.get('max_gap_bars', 0))}"
        )
    meta = {
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "data_source": data_source,
        "rows": int(len(frame)),
        "row_count": int(len(frame)),
        "load_mode": load_mode,
        "resolved_end_ts": str(resolved_end_ts),
        "resolved_end_required_effective": bool(resolved_end_required_effective),
        "resolved_end_missing": bool(resolved_end_missing),
        "canonical_scope_active": bool(frozen_mode),
        "continuity_report": continuity,
        "continuity_blocked": continuity_blocked,
        "continuity_reason": continuity_reason,
        "runtime_truth_blocked": runtime_truth_blocked,
        "runtime_truth_reason": runtime_truth_reason,
        "data_hash": stable_hash(
            {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "rows": int(len(frame)),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "resolved_end_ts": str(resolved_end_ts),
            },
            length=16,
        ),
        "continuity_policy_effective": {
            "strict_mode": strict_mode,
            "fail_on_gap": fail_on_gap,
            "frozen_research_mode": frozen_mode,
            "require_resolved_end_ts": require_resolved_end_ts,
        },
        "snapshot_meta": snapshot_metadata_from_config(config),
        "load_errors": load_errors,
    }
    return frame, meta


def run_candidate_replay(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
    frame: pd.DataFrame | None = None,
    market_meta: dict[str, Any] | None = None,
    perturbation_label: str = "base",
    cost_multiplier: float = 1.0,
    slippage_multiplier: float = 1.0,
    funding_multiplier: float = 1.0,
    hold_bars_multiplier: float = 1.0,
    signal_delay_bars: int = 0,
    signal_keep_ratio: float = 1.0,
) -> dict[str, Any]:
    """Run one actual replay/backtest for a Stage-52 candidate on real candles."""

    runtime_candidate = hydrate_candidate_record(candidate)
    timeframe = str(runtime_candidate.get("timeframe", "1h"))
    if frame is None or market_meta is None:
        frame, market_meta = load_candidate_market_frame(config, symbol=symbol, timeframe=timeframe)
    frame = frame.copy()
    market_meta = dict(market_meta or {})

    if frame.empty:
        return {
            "execution_status": "BLOCKED",
            "validation_state": "MISSING_MARKET_FRAME",
            "decision_use_allowed": False,
            "candidate": runtime_candidate,
            "market_meta": market_meta,
            "metrics": _empty_replay_metrics(),
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "backtest_params": {},
        }
    if bool(market_meta.get("runtime_truth_blocked", False)):
        return {
            "execution_status": "BLOCKED",
            "validation_state": str(market_meta.get("runtime_truth_reason", "RUNTIME_TRUTH_BLOCKED")),
            "decision_use_allowed": False,
            "candidate": runtime_candidate,
            "market_meta": market_meta,
            "metrics": _empty_replay_metrics(),
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "backtest_params": {},
        }
    if bool(market_meta.get("continuity_blocked", False)):
        return {
            "execution_status": "BLOCKED",
            "validation_state": "CONTINUITY_BLOCKED",
            "decision_use_allowed": False,
            "candidate": runtime_candidate,
            "market_meta": market_meta,
            "metrics": _empty_replay_metrics(),
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "backtest_params": {},
        }

    enriched = _prepare_runtime_frame(
        frame=frame,
        candidate=runtime_candidate,
        signal_delay_bars=int(max(0, signal_delay_bars)),
        signal_keep_ratio=float(max(0.05, min(signal_keep_ratio, 1.0))),
    )
    backtest_params = _candidate_backtest_params(
        frame=enriched,
        candidate=runtime_candidate,
        config=config,
        cost_multiplier=float(cost_multiplier),
        slippage_multiplier=float(slippage_multiplier),
        funding_multiplier=float(funding_multiplier),
        hold_bars_multiplier=float(hold_bars_multiplier),
    )
    result = run_backtest(
        frame=enriched,
        strategy_name=str(runtime_candidate.get("candidate_id", runtime_candidate.get("family", "candidate_runtime"))),
        symbol=str(symbol),
        signal_col="signal",
        atr_col="atr_14",
        initial_capital=float(backtest_params["initial_capital"]),
        stop_atr_multiple=float(backtest_params["stop_atr_multiple"]),
        take_profit_atr_multiple=float(backtest_params["take_profit_atr_multiple"]),
        max_hold_bars=int(backtest_params["max_hold_bars"]),
        round_trip_cost_pct=float(backtest_params["round_trip_cost_pct"]),
        slippage_pct=float(backtest_params["slippage_pct"]),
        funding_pct_per_day=float(backtest_params["funding_pct_per_day"]),
        exit_mode=str(backtest_params["exit_mode"]),
        trailing_atr_k=float(backtest_params["trailing_atr_k"]),
        partial_size=float(backtest_params["partial_size"]),
        position_sizing_mode=str(backtest_params["position_sizing_mode"]),
        risk_per_trade_pct=float(backtest_params["risk_per_trade_pct"]),
        fixed_fraction_pct=float(backtest_params["fixed_fraction_pct"]),
        cost_model_cfg=dict(config.get("cost_model", {})),
    )

    metrics = _replay_metrics_from_result(result)
    metrics["perturbation_label"] = str(perturbation_label)
    execution_status = "EXECUTED"
    validation_state = "REAL_REPLAY_READY" if int(metrics["trade_count"]) > 0 else "REAL_REPLAY_ZERO_TRADES"
    decision_use_allowed = bool(int(metrics["trade_count"]) > 0)
    return {
        "execution_status": execution_status,
        "validation_state": validation_state,
        "decision_use_allowed": decision_use_allowed,
        "candidate": runtime_candidate,
        "market_meta": market_meta,
        "metrics": metrics,
        "trades": result.trades.copy(),
        "equity_curve": result.equity_curve.copy(),
        "backtest_params": backtest_params,
    }


def evaluate_candidate_walkforward(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
    frame: pd.DataFrame | None = None,
    market_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run real time-based walk-forward validation for one executable candidate."""

    runtime_candidate = hydrate_candidate_record(candidate)
    timeframe = str(runtime_candidate.get("timeframe", "1h"))
    if frame is None or market_meta is None:
        frame, market_meta = load_candidate_market_frame(config, symbol=symbol, timeframe=timeframe)
    frame = frame.copy()
    market_meta = dict(market_meta or {})
    if frame.empty:
        return {
            "execution_status": "BLOCKED",
            "validation_state": "MISSING_MARKET_FRAME",
            "decision_use_allowed": False,
            "window_metrics": [],
            "summary": {
                "status": "PARTIAL",
                "classification": "INSUFFICIENT_DATA",
                "classification_explanation": "missing_market_frame",
                "usable_windows": 0,
                "total_windows": 0,
            },
            "forward_trades": pd.DataFrame(),
            "market_meta": market_meta,
            "candidate": runtime_candidate,
        }
    if bool(market_meta.get("runtime_truth_blocked", False)):
        return {
            "execution_status": "BLOCKED",
            "validation_state": str(market_meta.get("runtime_truth_reason", "RUNTIME_TRUTH_BLOCKED")),
            "decision_use_allowed": False,
            "window_metrics": [],
            "summary": {
                "status": "PARTIAL",
                "classification": "INSUFFICIENT_DATA",
                "classification_explanation": str(market_meta.get("runtime_truth_reason", "runtime_truth_blocked")).lower(),
                "usable_windows": 0,
                "total_windows": 0,
            },
            "forward_trades": pd.DataFrame(),
            "market_meta": market_meta,
            "candidate": runtime_candidate,
        }
    if bool(market_meta.get("continuity_blocked", False)):
        return {
            "execution_status": "BLOCKED",
            "validation_state": "CONTINUITY_BLOCKED",
            "decision_use_allowed": False,
            "window_metrics": [],
            "summary": {
                "status": "PARTIAL",
                "classification": "INSUFFICIENT_DATA",
                "classification_explanation": "continuity_blocked",
                "usable_windows": 0,
                "total_windows": 0,
            },
            "forward_trades": pd.DataFrame(),
            "market_meta": market_meta,
            "candidate": runtime_candidate,
        }

    enriched = _prepare_runtime_frame(frame=frame, candidate=runtime_candidate)
    wf_cfg = dict(config.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {}))
    windows = build_windows(
        start_ts=pd.to_datetime(enriched["timestamp"].min(), utc=True),
        end_ts=pd.to_datetime(enriched["timestamp"].max(), utc=True),
        train_days=int(max(30, wf_cfg.get("train_days", 180))),
        holdout_days=int(max(7, wf_cfg.get("holdout_days", 30))),
        forward_days=int(max(7, wf_cfg.get("forward_days", 30))),
        step_days=int(max(7, wf_cfg.get("step_days", 30))),
        reserve_tail_days=int(max(0, wf_cfg.get("reserve_tail_days", 0))),
    )
    params = _candidate_backtest_params(frame=enriched, candidate=runtime_candidate, config=config)
    window_metrics: list[dict[str, Any]] = []
    forward_trade_frames: list[pd.DataFrame] = []
    for window in windows:
        holdout = _slice_window(enriched, start=window.holdout_start, end=window.holdout_end)
        forward = _slice_window(enriched, start=window.forward_start, end=window.forward_end)
        holdout_result = _run_backtest_on_slice(holdout, params=params, candidate=runtime_candidate, symbol=symbol, config=config)
        forward_result = _run_backtest_on_slice(forward, params=params, candidate=runtime_candidate, symbol=symbol, config=config)
        holdout_metrics = _result_metrics(holdout_result, holdout)
        forward_metrics = _result_metrics(forward_result, forward)
        usable, reasons = _usable_forward_window(forward_metrics=forward_metrics, config=config)
        row = {
            "window_idx": int(window.window_idx),
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "holdout_start": window.holdout_start.isoformat(),
            "holdout_end": window.holdout_end.isoformat(),
            "forward_start": window.forward_start.isoformat(),
            "forward_end": window.forward_end.isoformat(),
            "train_bars": int(len(_slice_window(enriched, start=window.train_start, end=window.train_end))),
            "holdout_bars": int(len(holdout)),
            "forward_bars": int(len(forward)),
            "usable": bool(usable),
            "exclude_reasons": ";".join(reasons),
            "holdout_expectancy": float(holdout_metrics["expectancy"]),
            "holdout_profit_factor": float(holdout_metrics["profit_factor"]),
            "holdout_max_drawdown": float(holdout_metrics["max_drawdown"]),
            "holdout_return_pct": float(holdout_metrics["return_pct"]),
            "holdout_trade_count": int(holdout_metrics["trade_count"]),
            "holdout_exposure_ratio": float(holdout_metrics["exposure_ratio"]),
            "forward_expectancy": float(forward_metrics["expectancy"]),
            "forward_profit_factor": float(forward_metrics["profit_factor"]),
            "forward_max_drawdown": float(forward_metrics["max_drawdown"]),
            "forward_return_pct": float(forward_metrics["return_pct"]),
            "forward_trade_count": int(forward_metrics["trade_count"]),
            "forward_exposure_ratio": float(forward_metrics["exposure_ratio"]),
        }
        window_metrics.append(row)
        if usable and forward_result is not None and not forward_result.trades.empty:
            tagged = forward_result.trades.copy()
            tagged["window_idx"] = int(window.window_idx)
            forward_trade_frames.append(tagged)

    aggregate = aggregate_windows(window_metrics, config)
    promotion_cfg = dict(config.get("promotion_gates", {}).get("walkforward", {}))
    min_usable_windows = int(max(1, promotion_cfg.get("min_usable_windows", 3)))
    min_median_forward_exp_lcb = float(promotion_cfg.get("min_median_forward_exp_lcb", 0.0))
    median_expectancy = float(dict(aggregate.get("forward_expectancy", {})).get("median", 0.0))
    status = (
        "SUCCESS"
        if int(aggregate.get("usable_windows", 0)) >= min_usable_windows and median_expectancy >= min_median_forward_exp_lcb
        else "PARTIAL"
    )
    blocker_reason = "" if status == "SUCCESS" else "walkforward_gate_not_met"
    aggregate["status"] = status
    aggregate["median_forward_exp_lcb"] = median_expectancy
    aggregate["blocker_reason"] = blocker_reason
    aggregate["summary_hash"] = stable_hash(
        {
            "candidate_id": runtime_candidate.get("candidate_id", ""),
            "symbol": str(symbol),
            "timeframe": timeframe,
            "aggregate": aggregate,
        },
        length=16,
    )
    return {
        "execution_status": "EXECUTED",
        "validation_state": "REAL_VALIDATION_PASSED" if status == "SUCCESS" else "REAL_VALIDATION_FAILED",
        "decision_use_allowed": bool(status == "SUCCESS" or int(aggregate.get("usable_windows", 0)) > 0),
        "window_metrics": window_metrics,
        "summary": aggregate,
        "forward_trades": pd.concat(forward_trade_frames, ignore_index=True) if forward_trade_frames else pd.DataFrame(),
        "market_meta": market_meta,
        "candidate": runtime_candidate,
    }


def estimate_trade_monte_carlo(
    trades: pd.DataFrame,
    *,
    seed: int,
    n_paths: int,
    block_size: int,
) -> dict[str, Any]:
    """Estimate multi-scenario downside and drawdown stress from actual trade returns."""

    if trades.empty or "return_pct" not in trades.columns:
        return {
            "execution_status": "BLOCKED",
            "validation_state": "INSUFFICIENT_TRADES",
            "decision_use_allowed": False,
            "conservative_downside_bound": -1.0,
            "worst_case_drawdown_p95": 1.0,
            "n_paths": 0,
            "sample_size": 0,
            "block_size": int(block_size),
            "scenario_rows": [],
        }
    returns = pd.to_numeric(trades["return_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if int(len(returns)) < 8:
        return {
            "execution_status": "BLOCKED",
            "validation_state": "INSUFFICIENT_TRADES",
            "decision_use_allowed": False,
            "conservative_downside_bound": -1.0,
            "worst_case_drawdown_p95": 1.0,
            "n_paths": 0,
            "sample_size": int(len(returns)),
            "block_size": int(block_size),
            "scenario_rows": [],
        }
    rng = np.random.default_rng(int(seed))
    block = int(max(4, min(int(block_size), len(returns))))
    horizon = int(len(returns))
    scenarios = [
        {"label": "block_bootstrap", "mode": "block_bootstrap"},
        {"label": "trade_order_shuffle", "mode": "trade_order_shuffle"},
        {"label": "missed_trade_stress", "mode": "missed_trade_stress"},
        {"label": "cost_stress", "mode": "cost_stress"},
        {"label": "delay_stress", "mode": "delay_stress"},
    ]
    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        path_terminal_returns: list[float] = []
        path_drawdowns: list[float] = []
        for _ in range(int(max(100, n_paths))):
            sampled = _sample_trade_path(
                returns=returns,
                rng=rng,
                block_size=block,
                horizon=horizon,
                mode=str(scenario["mode"]),
            )
            equity = np.cumprod(1.0 + sampled)
            running_peak = np.maximum.accumulate(equity)
            drawdown = 1.0 - (equity / np.maximum(running_peak, 1e-9))
            path_terminal_returns.append(float(equity[-1] - 1.0))
            path_drawdowns.append(float(np.nanmax(drawdown)))
        terminal_p05 = float(np.quantile(np.asarray(path_terminal_returns, dtype=float), 0.05))
        drawdown_p95 = float(np.quantile(np.asarray(path_drawdowns, dtype=float), 0.95))
        scenario_rows.append(
            {
                "scenario": str(scenario["label"]),
                "terminal_return_p05": float(round(terminal_p05, 8)),
                "max_drawdown_p95": float(round(drawdown_p95, 8)),
                "passed": bool(terminal_p05 >= -0.02 and drawdown_p95 <= 0.35),
            }
        )
    downside = float(min(row["terminal_return_p05"] for row in scenario_rows))
    worst_drawdown = float(max(row["max_drawdown_p95"] for row in scenario_rows))
    return {
        "execution_status": "EXECUTED",
        "validation_state": "REAL_MONTE_CARLO_READY",
        "decision_use_allowed": bool(all(bool(row["passed"]) for row in scenario_rows)),
        "conservative_downside_bound": float(round(downside, 8)),
        "worst_case_drawdown_p95": float(round(worst_drawdown, 8)),
        "n_paths": int(max(100, n_paths)),
        "sample_size": int(len(returns)),
        "block_size": int(block),
        "scenario_rows": scenario_rows,
    }


def evaluate_cross_perturbation(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
    frame: pd.DataFrame | None = None,
    market_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run real perturbation robustness checks on one executable candidate."""

    runtime_candidate = hydrate_candidate_record(candidate)
    timeframe = str(runtime_candidate.get("timeframe", "1h"))
    if frame is None or market_meta is None:
        frame, market_meta = load_candidate_market_frame(config, symbol=symbol, timeframe=timeframe)
    market_meta = dict(market_meta or {})
    if frame.empty or bool(market_meta.get("continuity_blocked", False)) or bool(market_meta.get("runtime_truth_blocked", False)):
        return {
            "execution_status": "BLOCKED",
            "validation_state": str(market_meta.get("runtime_truth_reason", "INSUFFICIENT_RUNTIME_FRAME")) if bool(market_meta.get("runtime_truth_blocked", False)) else "INSUFFICIENT_RUNTIME_FRAME",
            "decision_use_allowed": False,
            "surviving_seeds": 0,
            "total_perturbations": 0,
            "rows": [],
        }

    perturbations = [
        {"label": "base", "cost_multiplier": 1.0, "slippage_multiplier": 1.0, "funding_multiplier": 1.0, "hold_bars_multiplier": 1.0, "signal_delay_bars": 0, "signal_keep_ratio": 1.0},
        {"label": "threshold_perturbation", "cost_multiplier": 1.0, "slippage_multiplier": 1.0, "funding_multiplier": 1.0, "hold_bars_multiplier": 1.0, "signal_delay_bars": 0, "signal_keep_ratio": 0.80},
        {"label": "cost_perturbation", "cost_multiplier": 1.5, "slippage_multiplier": 1.2, "funding_multiplier": 1.0, "hold_bars_multiplier": 1.0, "signal_delay_bars": 0, "signal_keep_ratio": 1.0},
        {"label": "timing_perturbation", "cost_multiplier": 1.0, "slippage_multiplier": 1.0, "funding_multiplier": 1.0, "hold_bars_multiplier": 1.0, "signal_delay_bars": 1, "signal_keep_ratio": 1.0},
        {"label": "feature_perturbation", "cost_multiplier": 1.0, "slippage_multiplier": 1.0, "funding_multiplier": 1.0, "hold_bars_multiplier": 0.85, "signal_delay_bars": 0, "signal_keep_ratio": 0.90},
    ]
    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    max_drawdown_gate = float(replay_gate.get("max_drawdown", 0.20))
    rows: list[dict[str, Any]] = []
    survivors = 0
    for perturb in perturbations:
        replay = run_candidate_replay(
            candidate=runtime_candidate,
            config=config,
            symbol=symbol,
            frame=frame,
            market_meta=market_meta,
            perturbation_label=str(perturb["label"]),
            cost_multiplier=float(perturb["cost_multiplier"]),
              slippage_multiplier=float(perturb["slippage_multiplier"]),
              funding_multiplier=float(perturb["funding_multiplier"]),
              hold_bars_multiplier=float(perturb["hold_bars_multiplier"]),
              signal_delay_bars=int(perturb["signal_delay_bars"]),
              signal_keep_ratio=float(perturb["signal_keep_ratio"]),
          )
        metrics = dict(replay.get("metrics", {}))
        passed = bool(
            replay.get("execution_status") == "EXECUTED"
            and int(metrics.get("trade_count", 0)) > 0
            and float(metrics.get("exp_lcb", -1.0)) >= 0.0
            and float(metrics.get("maxDD", 1.0)) <= max_drawdown_gate
        )
        if passed:
            survivors += 1
        rows.append(
            {
                "perturbation_label": str(perturb["label"]),
                "trade_count": int(metrics.get("trade_count", 0)),
                "exp_lcb": float(metrics.get("exp_lcb", 0.0)),
                  "maxDD": float(metrics.get("maxDD", 0.0)),
                  "passed": bool(passed),
                  "category": str(perturb["label"]).split("_")[0],
                  "execution_status": str(replay.get("execution_status", "BLOCKED")),
                  "validation_state": str(replay.get("validation_state", "")),
              }
          )
    return {
        "execution_status": "EXECUTED",
        "validation_state": "REAL_CROSS_PERTURBATION_READY",
        "decision_use_allowed": True,
        "surviving_seeds": int(survivors),
        "total_perturbations": int(len(rows)),
        "rows": rows,
    }


def compute_transfer_metrics(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    """Run a real transfer replay on a secondary symbol."""

    replay = run_candidate_replay(candidate=candidate, config=config, symbol=symbol)
    metrics = dict(replay.get("metrics", {}))
    return {
        "execution_status": str(replay.get("execution_status", "BLOCKED")),
        "validation_state": "REAL_TRANSFER_READY" if str(replay.get("execution_status", "")) == "EXECUTED" else str(replay.get("validation_state", "")),
        "decision_use_allowed": bool(replay.get("decision_use_allowed", False)),
        "trade_count": int(metrics.get("trade_count", 0)),
        "exp_lcb": float(metrics.get("exp_lcb", 0.0)),
        "maxDD": float(metrics.get("maxDD", 1.0)),
        "market_meta": dict(replay.get("market_meta", {})),
        "candidate": dict(replay.get("candidate", {})),
        "trades": replay.get("trades", pd.DataFrame()),
        "equity_curve": replay.get("equity_curve", pd.DataFrame()),
    }


def _prepare_runtime_frame(
    *,
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    signal_delay_bars: int = 0,
    signal_keep_ratio: float = 1.0,
) -> pd.DataFrame:
    bars = frame.copy().sort_values("timestamp").reset_index(drop=True)
    bars["timestamp"] = pd.to_datetime(bars.get("timestamp"), utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars.get(col), errors="coerce").astype(float)
    bars["atr_14"] = _atr14(bars)
    signal = build_candidate_signal_series(bars, candidate)
    extra_delay = int(max(0, signal_delay_bars))
    if extra_delay > 0:
        signal = signal.shift(extra_delay).fillna(0).astype(int)
    keep_ratio = float(max(0.05, min(signal_keep_ratio, 1.0)))
    if keep_ratio < 0.999:
        active_mask = signal.ne(0)
        if bool(active_mask.any()):
            active_positions = np.flatnonzero(active_mask.to_numpy())
            keep_count = max(1, int(np.floor(len(active_positions) * keep_ratio)))
            keep_positions = set(active_positions[:keep_count].tolist())
            values = signal.to_numpy(copy=True)
            for pos in active_positions:
                if int(pos) not in keep_positions:
                    values[int(pos)] = 0
            signal = pd.Series(values, index=signal.index, dtype=int)
    bars["signal"] = signal
    return bars


def build_candidate_signal_series(frame: pd.DataFrame, candidate: dict[str, Any]) -> pd.Series:
    """Convert one Stage-52 candidate family into an executable causal signal series."""

    signal_spec = _safe_mapping(candidate.get("signal_spec"))
    if str(signal_spec.get("type", "")).strip().lower() == "frame_columns":
        long_col = str(signal_spec.get("long_col", "")).strip()
        short_col = str(signal_spec.get("short_col", "")).strip()
        long_cond = _bool_series(frame[long_col]) if long_col and long_col in frame.columns else pd.Series(False, index=frame.index, dtype=bool)
        short_cond = _bool_series(frame[short_col]) if short_col and short_col in frame.columns else pd.Series(False, index=frame.index, dtype=bool)
        raw_signal = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        return pd.Series(raw_signal, index=frame.index, dtype=int).shift(1).fillna(0).astype(int)

    family = str(candidate.get("family", "")).strip()
    structure = compute_market_structure_engine(frame)
    liquidity = compute_liquidity_map(frame)
    volatility = compute_volatility_regime_engine(frame)
    flow = compute_flow_regime_engine(frame)
    htf_bias = structure["structural_bias"].map({"bull": "up", "bear": "down", "range": "flat"}).fillna("flat")
    mtf = compute_mtf_bias_completion(frame, htf_bias=htf_bias)
    geometry = compute_trade_geometry_layer(frame)

    close = pd.to_numeric(frame.get("close"), errors="coerce").fillna(0.0)
    open_ = pd.to_numeric(frame.get("open"), errors="coerce").fillna(close)
    long_cond = pd.Series(False, index=frame.index, dtype=bool)
    short_cond = pd.Series(False, index=frame.index, dtype=bool)

    if family == "structure_pullback_continuation":
        long_cond = (
            (structure["structural_bias"] == "bull")
            & structure["corrective_leg"]
            & flow["flow_confirmed_continuation"]
            & mtf["ltf_trigger_eligibility"]
            & geometry["invalidation_quality"]
        )
        short_cond = (
            (structure["structural_bias"] == "bear")
            & structure["corrective_leg"]
            & flow["flow_confirmed_continuation"]
            & mtf["ltf_trigger_eligibility"]
            & geometry["invalidation_quality"]
        )
    elif family == "liquidity_sweep_reversal":
        long_cond = (
            liquidity["liquidity_sweep_low"]
            & geometry["invalidation_quality"]
            & (flow["flow_exhaustion"] | (close > open_))
        )
        short_cond = (
            liquidity["liquidity_sweep_high"]
            & geometry["invalidation_quality"]
            & (flow["flow_exhaustion"] | (close < open_))
        )
    elif family == "squeeze_flow_breakout":
        long_cond = (
            volatility["breakout_readiness"]
            & flow["flow_burst"]
            & (flow["flow_imbalance"] > 0.0)
            & mtf["ltf_trigger_eligibility"]
        )
        short_cond = (
            volatility["breakout_readiness"]
            & flow["flow_burst"]
            & (flow["flow_imbalance"] < 0.0)
            & mtf["ltf_trigger_eligibility"]
        )
    elif family == "failed_breakout_reversal":
        long_cond = (
            liquidity["fake_breakout"]
            & liquidity["liquidity_sweep_low"]
            & flow["flow_exhaustion"]
            & geometry["invalidation_quality"]
        )
        short_cond = (
            liquidity["fake_breakout"]
            & liquidity["liquidity_sweep_high"]
            & flow["flow_exhaustion"]
            & geometry["invalidation_quality"]
        )
    elif family == "exhaustion_mean_reversion":
        mid = close.rolling(24, min_periods=1).mean().fillna(close)
        atr_source = frame["atr_14"] if "atr_14" in frame.columns else _atr14(frame)
        atr = pd.to_numeric(atr_source, errors="coerce").fillna(0.0)
        distance = (close - mid) / atr.replace(0.0, np.nan)
        long_cond = (distance <= -1.25).fillna(False) & flow["flow_exhaustion"]
        short_cond = (distance >= 1.25).fillna(False) & flow["flow_exhaustion"]
    elif family == "funding_oi_imbalance_reversion":
        crowding, _ = compute_crowding_layer(frame, timeframe=str(candidate.get("timeframe", "1h")))
        long_cond = crowding["sentiment_reversal_hint"] & (crowding["crowded_side_bias"] == "overshort")
        short_cond = crowding["sentiment_reversal_hint"] & (crowding["crowded_side_bias"] == "overlong")
    elif family == "volatility_regime_transition":
        long_cond = volatility["breakout_readiness"] & volatility["volatility_compression"].shift(1).fillna(False) & (flow["flow_imbalance"] > 0.0)
        short_cond = volatility["breakout_readiness"] & volatility["volatility_compression"].shift(1).fillna(False) & (flow["flow_imbalance"] < 0.0)
    elif family == "multi_tf_disagreement_repair":
        long_cond = mtf["structural_misalignment"] & structure["choch"] & (flow["flow_imbalance"] > 0.0)
        short_cond = mtf["structural_misalignment"] & structure["choch"] & (flow["flow_imbalance"] < 0.0)
    else:
        long_cond = structure["bos"] & (structure["structural_bias"] == "bull") & mtf["ltf_trigger_eligibility"]
        short_cond = structure["bos"] & (structure["structural_bias"] == "bear") & mtf["ltf_trigger_eligibility"]

    raw_signal = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return pd.Series(raw_signal, index=frame.index, dtype=int).shift(1).fillna(0).astype(int)


def _candidate_backtest_params(
    *,
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    config: dict[str, Any],
    cost_multiplier: float = 1.0,
    slippage_multiplier: float = 1.0,
    funding_multiplier: float = 1.0,
    hold_bars_multiplier: float = 1.0,
) -> dict[str, Any]:
    geometry = _safe_mapping(candidate.get("geometry"))
    atr_pct = (pd.to_numeric(frame["atr_14"], errors="coerce").fillna(0.0) / pd.to_numeric(frame["close"], errors="coerce").replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    median_atr_pct = float(atr_pct.median()) if not atr_pct.empty else 0.005
    stop_pct = float(max(1e-4, _safe_float(geometry.get("stop_distance_pct", 0.0060), default=0.0060)))
    target_pct = float(max(stop_pct * 1.2, _safe_float(geometry.get("first_target_pct", 0.0105), default=0.0105)))
    expected_hold_bars = int(max(2, _safe_int(geometry.get("expected_hold_bars", 12), default=12)))
    exit_mode = _candidate_exit_mode(candidate)
    risk_cfg = dict(config.get("risk", {}).get("sizing", {}))
    initial_equity = float(config.get("portfolio", {}).get("leverage_selector", {}).get("initial_equity", 10000.0))
    return {
        "initial_capital": initial_equity,
        "stop_atr_multiple": float(np.clip(stop_pct / max(median_atr_pct, 1e-6), 0.5, 8.0)),
        "take_profit_atr_multiple": float(np.clip(target_pct / max(median_atr_pct, 1e-6), 0.8, 12.0)),
        "max_hold_bars": int(max(2, round(expected_hold_bars * max(0.25, hold_bars_multiplier)))),
        "round_trip_cost_pct": float(config.get("costs", {}).get("round_trip_cost_pct", 0.1)) * float(max(0.1, cost_multiplier)),
        "slippage_pct": float(config.get("costs", {}).get("slippage_pct", 0.0005)) * float(max(0.1, slippage_multiplier)),
        "funding_pct_per_day": float(config.get("costs", {}).get("funding_pct_per_day", 0.0)) * float(max(0.1, funding_multiplier)),
        "exit_mode": exit_mode,
        "trailing_atr_k": 1.5 if exit_mode in {"trailing_atr", "partial_then_trail"} else 1.0,
        "partial_size": 0.5,
        "position_sizing_mode": str(risk_cfg.get("mode", "risk_budget")),
        "risk_per_trade_pct": _normalize_fraction(risk_cfg.get("risk_per_trade_pct", config.get("risk", {}).get("risk_per_trade_pct", 0.01)), default=0.01),
        "fixed_fraction_pct": _normalize_fraction(risk_cfg.get("fixed_fraction_pct", 0.10), default=0.10),
    }


def _candidate_exit_mode(candidate: dict[str, Any]) -> str:
    hold_logic = str(candidate.get("hold_logic", "")).lower()
    target_logic = str(candidate.get("target_logic", "")).lower()
    if "extension" in target_logic or "expansion" in target_logic:
        return "partial_then_trail"
    if "reversion" in target_logic:
        return "breakeven_1r"
    if "hold_while" in hold_logic:
        return "trailing_atr"
    return "fixed_atr"


def _replay_metrics_from_result(result: Any) -> dict[str, Any]:
    trades = result.trades.copy() if result is not None else pd.DataFrame()
    trade_count = int(len(trades))
    trade_returns = (
        pd.to_numeric(trades.get("return_pct", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not trades.empty
        else pd.Series(dtype=float)
    )
    if trade_returns.empty:
        exp_lcb = -1.0
    else:
        std = float(trade_returns.std(ddof=0)) if int(len(trade_returns)) > 1 else 0.0
        mean = float(trade_returns.mean())
        exp_lcb = float(mean - (1.28155 * std / max(len(trade_returns) ** 0.5, 1.0)))
    failure_reason_dominance = (
        float(trades.get("exit_reason", pd.Series(dtype=str)).astype(str).value_counts(normalize=True).max())
        if trade_count > 0 and "exit_reason" in trades.columns
        else 1.0
    )
    return {
        "trade_count": int(trade_count),
        "exp_lcb": float(round(exp_lcb, 8)),
        "maxDD": float(round(float(result.metrics.get("max_drawdown", 1.0)) if result is not None else 1.0, 8)),
        "failure_reason_dominance": float(round(failure_reason_dominance, 8)),
        "expectancy": float(round(float(result.metrics.get("expectancy", 0.0)) if result is not None else 0.0, 8)),
        "profit_factor": float(round(float(result.metrics.get("profit_factor", 0.0)) if result is not None else 0.0, 8)),
    }


def _run_backtest_on_slice(
    slice_frame: pd.DataFrame,
    *,
    params: dict[str, Any],
    candidate: dict[str, Any],
    symbol: str,
    config: dict[str, Any],
) -> Any | None:
    if slice_frame.empty:
        return None
    return run_backtest(
        frame=slice_frame,
        strategy_name=str(candidate.get("candidate_id", candidate.get("family", "candidate_runtime"))),
        symbol=str(symbol),
        signal_col="signal",
        atr_col="atr_14",
        initial_capital=float(params["initial_capital"]),
        stop_atr_multiple=float(params["stop_atr_multiple"]),
        take_profit_atr_multiple=float(params["take_profit_atr_multiple"]),
        max_hold_bars=int(params["max_hold_bars"]),
        round_trip_cost_pct=float(params["round_trip_cost_pct"]),
        slippage_pct=float(params["slippage_pct"]),
        funding_pct_per_day=float(params["funding_pct_per_day"]),
        exit_mode=str(params["exit_mode"]),
        trailing_atr_k=float(params["trailing_atr_k"]),
        partial_size=float(params["partial_size"]),
        position_sizing_mode=str(params["position_sizing_mode"]),
        risk_per_trade_pct=float(params["risk_per_trade_pct"]),
        fixed_fraction_pct=float(params["fixed_fraction_pct"]),
        cost_model_cfg=dict(config.get("cost_model", {})),
    )


def _result_metrics(result: Any | None, frame: pd.DataFrame) -> dict[str, float]:
    if result is None or frame.empty:
        return {
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "return_pct": 0.0,
            "trade_count": 0.0,
            "exposure_ratio": 0.0,
            "finite": True,
        }
    trade_count = float(result.metrics.get("trade_count", 0.0))
    bars = max(1, len(frame))
    exposure_ratio = 0.0
    if not result.trades.empty and "bars_held" in result.trades.columns:
        exposure_ratio = float(pd.to_numeric(result.trades["bars_held"], errors="coerce").fillna(0.0).sum() / float(bars))
    final_equity = float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else 0.0
    initial_equity = float(result.equity_curve["equity"].iloc[0]) if not result.equity_curve.empty else 1.0
    return_pct = float((final_equity / initial_equity) - 1.0) if initial_equity != 0 else 0.0
    out = {
        "expectancy": float(result.metrics.get("expectancy", 0.0)),
        "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
        "return_pct": float(return_pct),
        "trade_count": float(trade_count),
        "exposure_ratio": float(exposure_ratio),
    }
    out["finite"] = all(np.isfinite(float(v)) for v in out.values())
    return out


def _usable_forward_window(*, forward_metrics: dict[str, float], config: dict[str, Any]) -> tuple[bool, list[str]]:
    portfolio_cfg = dict(config.get("portfolio", {}).get("walkforward", {}))
    min_trades = int(max(1, portfolio_cfg.get("min_forward_trades", 10)))
    min_exposure = float(max(0.0, portfolio_cfg.get("min_forward_exposure", 0.01)))
    reasons: list[str] = []
    if not bool(forward_metrics.get("finite", False)):
        reasons.append("non_finite_metrics")
    if float(forward_metrics.get("trade_count", 0.0)) < float(min_trades):
        reasons.append("min_trades")
    if float(forward_metrics.get("exposure_ratio", 0.0)) < float(min_exposure):
        reasons.append("min_exposure")
    return len(reasons) == 0, reasons


def _slice_window(frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    mask = (ts >= pd.to_datetime(start, utc=True)) & (ts < pd.to_datetime(end, utc=True))
    return frame.loc[mask].copy().reset_index(drop=True)


def _atr14(frame: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(frame.get("high"), errors="coerce").astype(float)
    low = pd.to_numeric(frame.get("low"), errors="coerce").astype(float)
    close = pd.to_numeric(frame.get("close"), errors="coerce").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14, min_periods=3).mean().fillna(tr.expanding().mean()).replace(0.0, 1e-6)


def _sample_trade_path(
    *,
    returns: np.ndarray,
    rng: np.random.Generator,
    block_size: int,
    horizon: int,
    mode: str,
) -> np.ndarray:
    clean = np.asarray(returns, dtype=float)
    if str(mode) == "trade_order_shuffle":
        sample = rng.permutation(clean.copy())
    else:
        idx = int(rng.integers(0, max(1, len(clean) - block_size + 1)))
        segment = clean[idx : idx + block_size]
        reps = int(max(1, np.ceil(horizon / max(1, len(segment)))))
        sample = np.tile(segment, reps)[:horizon]
    if str(mode) == "missed_trade_stress":
        miss_mask = np.arange(len(sample)) % 5 == 0
        sample = sample.copy()
        sample[miss_mask] = 0.0
    elif str(mode) == "cost_stress":
        sample = sample.copy() - 0.00075
    elif str(mode) == "delay_stress":
        sample = np.where(sample > 0.0, sample * 0.80, sample * 1.10)
    return sample.astype(float)


def _safe_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = ast.literal_eval(text)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _generic_runtime_geometry(*, family: str, timeframe: str) -> dict[str, Any]:
    scale_map = {"15m": 0.75, "30m": 0.85, "1h": 1.0, "2h": 1.15, "4h": 1.35}
    scale = float(scale_map.get(str(timeframe), 1.0))
    stop_pct = 0.0055 * scale
    first_target_pct = 0.0105 * scale
    stretch_target_pct = 0.0160 * scale
    return {
        "entry_zone": {"low": 0.99, "high": 1.01},
        "invalidation_event": f"generic_invalidation::{family or 'unknown'}",
        "stop_distance_pct": float(round(stop_pct, 8)),
        "first_target_pct": float(round(first_target_pct, 8)),
        "stretch_target_pct": float(round(stretch_target_pct, 8)),
        "expected_hold_bars": int(max(4, round(12 * scale))),
        "estimated_round_trip_cost_pct": 0.001,
        "entry_logic": "generic_context_trigger_entry",
        "stop_logic": "generic_structure_invalidation",
        "target_logic": "generic_two_stage_exit",
        "hold_logic": "generic_time_or_invalidation_exit",
    }


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _bool_series(raw: Any) -> pd.Series:
    series = pd.Series(raw) if not isinstance(raw, pd.Series) else raw.copy()
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).astype(float) > 0.0
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y", "long", "short"})


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _normalize_fraction(value: Any, *, default: float) -> float:
    numeric = _safe_float(value, default=default)
    if numeric > 1.0:
        numeric = numeric / 100.0
    return float(max(0.0, min(numeric, 1.0)))


def _empty_replay_metrics() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "exp_lcb": -1.0,
        "maxDD": 1.0,
        "failure_reason_dominance": 1.0,
        "expectancy": 0.0,
        "profit_factor": 0.0,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
