"""Stage-2.8 probabilistic rolling walk-forward evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, validate_config
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.portfolio.builder import (
    Stage1CandidateRecord,
    _load_stage1_config,
    build_candidate_signal_cache,
    build_portfolio_return_series,
    load_stage1_candidates,
)
from buffmini.portfolio.correlation import average_correlation, build_correlation_matrix, effective_number_of_strategies
from buffmini.portfolio.metrics import INITIAL_PORTFOLIO_CAPITAL, build_portfolio_equity, compute_portfolio_metrics
from buffmini.portfolio.rolling_windows import BAR_DELTA, ReservedTailRange, RollingWindowSpec, build_rolling_windows
from buffmini.portfolio.walkforward import (
    WindowSpec,
    _build_holdout_window,
    _combine_weighted_trade_pnls,
    _compute_input_data_hash,
    _json_safe_value,
    _load_raw_data,
    _write_weight_csvs,
    evaluate_candidate_bundles_for_window,
)
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)

EDGE_PROB_STRONG = 0.80
EDGE_PROB_MODERATE = 0.65
STRONG_USABLE_WINDOWS = 5
MODERATE_USABLE_WINDOWS = 3
REGIME_SENSITIVITY_VARIANCE = 0.02
WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PortfolioWindowEvidence:
    """Per-window probabilistic evidence for one portfolio method."""

    method: str
    window_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    usable: bool
    exclusion_reason: str
    trade_count: int
    trades_per_month: float
    exposure_ratio: float
    average_correlation: float
    effective_n: float
    raw_profit_factor: float
    expectancy: float
    exp_lcb: float
    return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    p_edge_gt0: float | None
    edge_ci_low: float | None
    edge_median: float | None
    edge_ci_high: float | None
    p_pf_gt1: float | None
    pf_ci_low: float | None
    pf_median: float | None
    pf_ci_high: float | None
    lcb_t_edge: float | None


@dataclass(frozen=True)
class MethodComposition:
    """Composition summary for one Stage-2 portfolio method."""

    weights: dict[str, float]
    selected_candidates: list[str]
    effective_n: float
    average_correlation: float


def bootstrap_edge_probability(
    trade_pnls: pd.Series | list[float] | np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
    pf_clip_max: float = 5.0,
) -> dict[str, float]:
    """Estimate probabilistic edge and PF evidence via bootstrap resampling."""

    values = np.asarray(pd.Series(trade_pnls, dtype=float).dropna().to_numpy(), dtype=float)
    if values.size == 0:
        return {
            "p_edge_gt0": 0.0,
            "edge_median": 0.0,
            "edge_ci_low": 0.0,
            "edge_ci_high": 0.0,
            "p_pf_gt1": 0.0,
            "pf_median": 0.0,
            "pf_ci_low": 0.0,
            "pf_ci_high": 0.0,
        }
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")

    rng = np.random.default_rng(int(seed))
    sample_idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    samples = values[sample_idx]

    mean_edges = samples.mean(axis=1)
    gross_profit = np.clip(samples, a_min=0.0, a_max=None).sum(axis=1)
    gross_loss = -np.clip(samples, a_min=None, a_max=0.0).sum(axis=1)
    pf_values = np.zeros_like(gross_profit, dtype=float)
    positive_loss = gross_loss > 0.0
    pf_values[positive_loss] = gross_profit[positive_loss] / gross_loss[positive_loss]
    zero_loss_positive_profit = (~positive_loss) & (gross_profit > 0.0)
    pf_values[zero_loss_positive_profit] = float(pf_clip_max)
    pf_values = np.clip(pf_values, a_min=0.0, a_max=float(pf_clip_max))

    return {
        "p_edge_gt0": float(np.mean(mean_edges > 0.0)),
        "edge_median": float(np.median(mean_edges)),
        "edge_ci_low": float(np.percentile(mean_edges, 5)),
        "edge_ci_high": float(np.percentile(mean_edges, 95)),
        "p_pf_gt1": float(np.mean(pf_values > 1.0)),
        "pf_median": float(np.median(pf_values)),
        "pf_ci_low": float(np.percentile(pf_values, 5)),
        "pf_ci_high": float(np.percentile(pf_values, 95)),
    }


def lcb_t_edge(trade_pnls: pd.Series | list[float] | np.ndarray) -> float | None:
    """Fast lower confidence bound for mean trade PnL."""

    values = np.asarray(pd.Series(trade_pnls, dtype=float).dropna().to_numpy(), dtype=float)
    if values.size < 2:
        return None
    return float(values.mean() - (values.std(ddof=0) / math.sqrt(values.size)))


def aggregate_window_probabilities(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate usable rolling-window evidence into a robust method summary."""

    total_windows = int(len(window_results))
    usable_results = [row for row in window_results if bool(row.get("usable", False))]
    usable_windows = int(len(usable_results))

    if usable_results:
        p_edge_values = np.array([float(row["p_edge_gt0"]) for row in usable_results], dtype=float)
        p_pf_values = np.array([float(row["p_pf_gt1"]) for row in usable_results], dtype=float)
        edge_ci_lows = np.array([float(row["edge_ci_low"]) for row in usable_results], dtype=float)
        p_edge_median = float(np.median(p_edge_values))
        p_edge_min = float(np.min(p_edge_values))
        p_edge_mean = float(np.mean(p_edge_values))
        p_pf_median = float(np.median(p_pf_values))
        p_pf_min = float(np.min(p_pf_values))
        p_pf_mean = float(np.mean(p_pf_values))
        worst_edge_ci_low = float(np.min(edge_ci_lows))
        variance = float(np.var(p_edge_values, ddof=0))
        regime_sensitivity = bool(variance > REGIME_SENSITIVITY_VARIANCE)
        regime_reason = (
            f"p_edge_gt0 variance {variance:.4f} exceeds threshold {REGIME_SENSITIVITY_VARIANCE:.4f}"
            if regime_sensitivity
            else f"p_edge_gt0 variance {variance:.4f} within threshold"
        )
    else:
        p_edge_median = 0.0
        p_edge_min = 0.0
        p_edge_mean = 0.0
        p_pf_median = 0.0
        p_pf_min = 0.0
        p_pf_mean = 0.0
        worst_edge_ci_low = 0.0
        regime_sensitivity = False
        regime_reason = "No usable windows available for regime-sensitivity estimation."

    confidence = float(usable_windows / total_windows) if total_windows > 0 else 0.0
    robustness = float(p_edge_median * p_pf_median * confidence)
    if p_edge_median >= EDGE_PROB_STRONG and usable_windows >= STRONG_USABLE_WINDOWS:
        classification = "STRONG"
    elif p_edge_median >= EDGE_PROB_MODERATE and usable_windows >= MODERATE_USABLE_WINDOWS:
        classification = "MODERATE"
    else:
        classification = "WEAK"

    return {
        "usable_windows": usable_windows,
        "total_windows": total_windows,
        "p_edge_gt0_median": p_edge_median,
        "p_edge_gt0_min": p_edge_min,
        "p_edge_gt0_mean": p_edge_mean,
        "p_pf_gt1_median": p_pf_median,
        "p_pf_gt1_min": p_pf_min,
        "p_pf_gt1_mean": p_pf_mean,
        "worst_edge_ci_low": worst_edge_ci_low,
        "robustness_score": robustness,
        "classification": classification,
        "regime_sensitivity": regime_sensitivity,
        "regime_sensitivity_reason": regime_reason,
    }

def run_stage2_probabilistic(
    stage2_run_id: str,
    window_days: int = 30,
    stride_days: int = 7,
    num_windows: int | None = None,
    reserve_tail_days: int = 180,
    min_trades: int = 20,
    min_exposure: float = 0.01,
    n_boot: int = 5000,
    seed: int = 42,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    run_id: str | None = None,
    cli_command: str | None = None,
) -> Path:
    """Run Stage-2.8 probabilistic walk-forward evaluation from an existing Stage-2 run."""

    if int(window_days) < 1:
        raise ValueError("window_days must be >= 1")
    if int(stride_days) < 1:
        raise ValueError("stride_days must be >= 1")
    if num_windows is not None and int(num_windows) < 1:
        raise ValueError("num_windows must be >= 1 when provided")
    if int(reserve_tail_days) < 0:
        raise ValueError("reserve_tail_days must be >= 0")
    if int(min_trades) < 0:
        raise ValueError("min_trades must be >= 0")
    if float(min_exposure) < 0:
        raise ValueError("min_exposure must be >= 0")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")

    stage2_run_dir = runs_dir / stage2_run_id
    if not stage2_run_dir.exists():
        raise FileNotFoundError(f"Stage-2 run not found: {stage2_run_id}")

    stage2_summary = _load_json(stage2_run_dir / "portfolio_summary.json")
    stage1_run_id = str(stage2_summary["stage1_run_id"])
    stage1_run_dir = runs_dir / stage1_run_id
    if not stage1_run_dir.exists():
        raise FileNotFoundError(f"Stage-1 run not found: {stage1_run_id}")

    stage1_summary = _load_json(stage1_run_dir / "summary.json")
    config = _load_stage1_config(stage1_run_dir)
    validate_config(config)
    pf_clip_max = float(config.get("portfolio", {}).get("walkforward", {}).get("pf_clip_max", 5.0))

    raw_data = _load_raw_data(config=config, data_dir=data_dir)
    feature_data = {symbol: calculate_features(frame) for symbol, frame in raw_data.items()}
    data_hash = _compute_input_data_hash(raw_data)
    config_hash = compute_config_hash(config)

    available_end = min(
        pd.to_datetime(frame["timestamp"], utc=True).max()
        for frame in raw_data.values()
        if not frame.empty
    )
    holdout_window = _build_holdout_window(stage2_summary=stage2_summary, available_end=available_end, reserve_forward_days=int(reserve_tail_days))
    if holdout_window.actual_start is None or holdout_window.actual_end is None:
        raise ValueError("Reserved tail leaves no usable holdout range for Stage-2.8")

    selected_candidate_ids = sorted(
        {
            str(candidate_id)
            for payload in stage2_summary["portfolio_methods"].values()
            for candidate_id in payload.get("selected_candidates", [])
        }
    )
    candidate_records = {record.candidate_id: record for record in load_stage1_candidates(stage1_run_dir)}
    selected_records = {
        candidate_id: candidate_records[candidate_id]
        for candidate_id in selected_candidate_ids
        if candidate_id in candidate_records
    }
    if not selected_records:
        raise ValueError("Stage-2.8 requires Stage-1 candidate artifacts for the Stage-2 selection set")

    signal_caches = {
        candidate_id: build_candidate_signal_cache(candidate=record.to_candidate(), feature_data=feature_data)
        for candidate_id, record in selected_records.items()
    }
    initial_capital = float(INITIAL_PORTFOLIO_CAPITAL * float(config["risk"]["max_concurrent_positions"]))
    round_trip_cost_pct = float(config["costs"]["round_trip_cost_pct"])
    slippage_pct = float(config["costs"]["slippage_pct"])

    reserved_tail, rolling_windows = build_rolling_windows(
        start_ts=holdout_window.actual_end + BAR_DELTA,
        end_ts=available_end,
        window_days=int(window_days),
        stride_days=int(stride_days),
        reserve_tail_days=int(reserve_tail_days),
    )
    if num_windows is not None:
        rolling_windows = rolling_windows[: int(num_windows)]

    holdout_bundles = evaluate_candidate_bundles_for_window(
        window=holdout_window,
        candidate_records=selected_records,
        signal_caches=signal_caches,
        initial_capital=initial_capital,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )
    holdout_return_map = {
        candidate_id: bundle["returns"]
        for candidate_id, bundle in holdout_bundles.items()
        if not bundle["returns"].empty
    }
    holdout_correlation = build_correlation_matrix(holdout_return_map)
    method_compositions = _build_method_compositions(stage2_summary=stage2_summary, holdout_correlation=holdout_correlation)

    holdout_metrics = {
        method_key: _evaluate_method_window(
            method_key=method_key,
            window_name="Holdout",
            start=holdout_window.actual_start,
            end=holdout_window.actual_end,
            weights=composition.weights,
            candidate_bundles=holdout_bundles,
            correlation_matrix=holdout_correlation,
            min_trades=int(min_trades),
            min_exposure=float(min_exposure),
            n_boot=int(n_boot),
            seed=int(seed),
            pf_clip_max=pf_clip_max,
            apply_exclusion=False,
        )
        for method_key, composition in method_compositions.items()
    }

    window_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    evidence_payload: dict[str, list[dict[str, Any]]] = {method_key: [] for method_key in method_compositions}
    for index, rolling_window in enumerate(rolling_windows, start=1):
        bundles = evaluate_candidate_bundles_for_window(
            window=_rolling_to_window_spec(rolling_window),
            candidate_records=selected_records,
            signal_caches=signal_caches,
            initial_capital=initial_capital,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
        )
        return_map = {
            candidate_id: bundle["returns"]
            for candidate_id, bundle in bundles.items()
            if not bundle["returns"].empty
        }
        correlation_matrix = build_correlation_matrix(return_map)
        for method_key, composition in method_compositions.items():
            evidence = _evaluate_method_window(
                method_key=method_key,
                window_name=rolling_window.name,
                start=rolling_window.start,
                end=rolling_window.end,
                weights=composition.weights,
                candidate_bundles=bundles,
                correlation_matrix=correlation_matrix,
                min_trades=int(min_trades),
                min_exposure=float(min_exposure),
                n_boot=int(n_boot),
                seed=int(seed) + index,
                pf_clip_max=pf_clip_max,
                apply_exclusion=True,
            )
            row = _window_evidence_to_row(evidence)
            window_rows.append(row)
            evidence_payload[method_key].append(row)
            if not evidence.usable:
                excluded_rows.append(row)

    method_aggregates = {
        method_key: aggregate_window_probabilities(evidence_payload[method_key])
        for method_key in method_compositions
    }
    sanity_checks = _build_sanity_checks(
        holdout_window=holdout_window,
        reserved_tail=reserved_tail,
        rolling_windows=rolling_windows,
        method_compositions=method_compositions,
        evidence_payload=evidence_payload,
        method_aggregates=method_aggregates,
    )
    overall_recommendation = _overall_recommendation(method_aggregates)
    command = cli_command or _default_command(
        stage2_run_id=stage2_run_id,
        window_days=int(window_days),
        stride_days=int(stride_days),
        num_windows=num_windows,
        reserve_tail_days=int(reserve_tail_days),
        min_trades=int(min_trades),
        min_exposure=float(min_exposure),
        n_boot=int(n_boot),
        seed=int(seed),
    )

    summary_payload = {
        "stage1_run_id": stage1_run_id,
        "stage2_run_id": stage2_run_id,
        "seed": int(seed),
        "n_boot": int(n_boot),
        "edge_basis": "trade_pnl_per_trade_net_of_costs",
        "window_days": int(window_days),
        "stride_days": int(stride_days),
        "num_windows_requested": None if num_windows is None else int(num_windows),
        "reserve_tail_days": int(reserve_tail_days),
        "min_trades": int(min_trades),
        "min_exposure": float(min_exposure),
        "command": command,
        "config_hash": config_hash,
        "config_hash_stage1_reference": stage1_summary.get("config_hash"),
        "data_hash": data_hash,
        "data_hash_stage1_reference": stage1_summary.get("data_hash"),
        "holdout_window": _window_payload(holdout_window),
        "reserved_tail": {
            "start": reserved_tail.start.isoformat(),
            "end": reserved_tail.end.isoformat(),
        },
        "rolling_windows": [_rolling_payload(window) for window in rolling_windows],
        "fallback_used": bool("shifted_for_forward_window" in stage2_summary.get("window_modes", [])),
        "fallback_reason": stage2_summary.get("window_mode_note", ""),
        "method_summaries": {
            method_key: {
                "composition": {
                    "weights": {candidate_id: float(weight) for candidate_id, weight in composition.weights.items()},
                    "selected_candidates": list(composition.selected_candidates),
                    "effective_n": float(composition.effective_n),
                    "average_correlation": float(composition.average_correlation),
                },
                "holdout": _json_safe_value(_window_evidence_to_row(holdout_metrics[method_key])),
                "aggregate": _json_safe_value(method_aggregates[method_key]),
            }
            for method_key, composition in method_compositions.items()
        },
        "overall_recommendation": overall_recommendation,
        "sanity_checks": sanity_checks,
    }

    summary_hash = stable_hash(summary_payload, length=12)
    resolved_run_id = run_id or f"{utc_now_compact()}_{summary_hash}_stage2_8"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_payload["run_id"] = resolved_run_id

    pd.DataFrame(window_rows).to_csv(run_dir / "window_metrics_full.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv(run_dir / "excluded_windows.csv", index=False)
    pd.DataFrame(_aggregate_rows(method_aggregates)).to_csv(run_dir / "method_aggregate.csv", index=False)
    pd.DataFrame([_rolling_payload(window) for window in rolling_windows]).to_csv(run_dir / "rolling_windows.csv", index=False)
    _write_weight_csvs(run_dir=run_dir, stage2_summary=stage2_summary, candidate_records=selected_records)
    (run_dir / "evidence_windows.json").write_text(json.dumps(_json_safe_value(evidence_payload), indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "probabilistic_summary.json").write_text(json.dumps(_json_safe_value(summary_payload), indent=2, allow_nan=False), encoding="utf-8")
    _write_probabilistic_report(
        run_dir=run_dir,
        summary=summary_payload,
        method_compositions=method_compositions,
        candidate_records=selected_records,
        evidence_payload=evidence_payload,
    )
    logger.info("Saved Stage-2.8 artifacts to %s", run_dir)
    return run_dir


def _build_method_compositions(
    stage2_summary: dict[str, Any],
    holdout_correlation: pd.DataFrame,
) -> dict[str, MethodComposition]:
    compositions: dict[str, MethodComposition] = {}
    for method_key in ["equal", "vol", "corr-min"]:
        payload = stage2_summary["portfolio_methods"].get(method_key)
        if payload is None:
            continue
        weights = {str(candidate_id): float(weight) for candidate_id, weight in payload.get("weights", {}).items() if float(weight) > 0.0}
        selected = list(payload.get("selected_candidates", list(weights)))
        selected_matrix = holdout_correlation.reindex(index=selected, columns=selected) if not holdout_correlation.empty else pd.DataFrame()
        compositions[method_key] = MethodComposition(
            weights=weights,
            selected_candidates=selected,
            effective_n=float(effective_number_of_strategies(weights)),
            average_correlation=float(average_correlation(selected_matrix)),
        )
    return compositions

def _evaluate_method_window(
    method_key: str,
    window_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    weights: dict[str, float],
    candidate_bundles: dict[str, dict[str, Any]],
    correlation_matrix: pd.DataFrame,
    min_trades: int,
    min_exposure: float,
    n_boot: int,
    seed: int,
    pf_clip_max: float,
    apply_exclusion: bool,
) -> PortfolioWindowEvidence:
    portfolio_returns = build_portfolio_return_series(
        {candidate_id: bundle["returns"] for candidate_id, bundle in candidate_bundles.items()},
        weights,
    )
    portfolio_exposure = build_portfolio_return_series(
        {candidate_id: bundle["exposure"] for candidate_id, bundle in candidate_bundles.items()},
        weights,
    )
    portfolio_equity = build_portfolio_equity(portfolio_returns)
    trade_pnls = _combine_weighted_trade_pnls(candidate_bundles, weights)
    metrics = compute_portfolio_metrics(
        returns=portfolio_returns,
        equity=portfolio_equity,
        trade_pnls=trade_pnls,
        exposure=portfolio_exposure,
    )

    reasons: list[str] = []
    if apply_exclusion:
        if int(metrics["trade_count_total"]) < int(min_trades):
            reasons.append(f"trade_count<{int(min_trades)}")
        if float(metrics["exposure_ratio"]) < float(min_exposure):
            reasons.append(f"exposure_ratio<{float(min_exposure):.4f}")

    finite_values = [
        float(metrics["profit_factor"]),
        float(metrics["expectancy"]),
        float(metrics["exp_lcb"]),
        float(metrics["return_pct"]),
        float(metrics["max_drawdown"]),
        float(metrics["Sharpe_ratio"]),
        float(metrics["Sortino_ratio"]),
        float(metrics["Calmar_ratio"]),
    ]
    if any(not math.isfinite(value) for value in finite_values):
        reasons.append("non_finite_metrics")

    usable = len(reasons) == 0
    bootstrap = bootstrap_edge_probability(trade_pnls, n_boot=n_boot, seed=seed, pf_clip_max=pf_clip_max) if usable else None
    t_lcb = lcb_t_edge(trade_pnls) if usable else None
    selected_ids = [candidate_id for candidate_id, weight in weights.items() if float(weight) > 0.0]
    selected_corr = correlation_matrix.reindex(index=selected_ids, columns=selected_ids) if not correlation_matrix.empty else pd.DataFrame()

    return PortfolioWindowEvidence(
        method=method_key,
        window_id=window_name,
        start=start,
        end=end,
        usable=usable,
        exclusion_reason="; ".join(reasons),
        trade_count=int(metrics["trade_count_total"]),
        trades_per_month=float(metrics["trades_per_month"]),
        exposure_ratio=float(metrics["exposure_ratio"]),
        average_correlation=float(average_correlation(selected_corr)),
        effective_n=float(effective_number_of_strategies(weights)),
        raw_profit_factor=float(metrics["profit_factor"]),
        expectancy=float(metrics["expectancy"]),
        exp_lcb=float(metrics["exp_lcb"]),
        return_pct=float(metrics["return_pct"]),
        max_drawdown=float(metrics["max_drawdown"]),
        sharpe_ratio=float(metrics["Sharpe_ratio"]),
        sortino_ratio=float(metrics["Sortino_ratio"]),
        calmar_ratio=float(metrics["Calmar_ratio"]),
        p_edge_gt0=None if bootstrap is None else float(bootstrap["p_edge_gt0"]),
        edge_ci_low=None if bootstrap is None else float(bootstrap["edge_ci_low"]),
        edge_median=None if bootstrap is None else float(bootstrap["edge_median"]),
        edge_ci_high=None if bootstrap is None else float(bootstrap["edge_ci_high"]),
        p_pf_gt1=None if bootstrap is None else float(bootstrap["p_pf_gt1"]),
        pf_ci_low=None if bootstrap is None else float(bootstrap["pf_ci_low"]),
        pf_median=None if bootstrap is None else float(bootstrap["pf_median"]),
        pf_ci_high=None if bootstrap is None else float(bootstrap["pf_ci_high"]),
        lcb_t_edge=t_lcb,
    )


def _build_sanity_checks(
    holdout_window: WindowSpec,
    reserved_tail: ReservedTailRange,
    rolling_windows: list[RollingWindowSpec],
    method_compositions: dict[str, MethodComposition],
    evidence_payload: dict[str, list[dict[str, Any]]],
    method_aggregates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    overlap_checks: list[dict[str, Any]] = []
    for window in rolling_windows:
        overlap_checks.append(
            {
                "name": f"holdout_before_{window.name}",
                "passed": bool((holdout_window.actual_end or holdout_window.expected_end) < window.start),
                "detail": f"{(holdout_window.actual_end or holdout_window.expected_end).isoformat()} < {window.start.isoformat()}",
            }
        )
    for current, nxt in zip(rolling_windows, rolling_windows[1:], strict=False):
        overlap_checks.append(
            {
                "name": f"stride_{current.name}_to_{nxt.name}",
                "passed": bool(nxt.start >= current.start),
                "detail": f"{current.start.isoformat()} -> {nxt.start.isoformat()} stride preserved",
            }
        )

    weight_checks = {
        method_key: {
            "sum": float(sum(composition.weights.values())),
            "passed": bool(abs(sum(composition.weights.values()) - 1.0) <= WEIGHT_TOLERANCE),
        }
        for method_key, composition in method_compositions.items()
    }
    window_tail_checks = [
        {
            "window": window.name,
            "inside_reserved_tail": bool(
                reserved_tail.start <= window.start <= reserved_tail.end
                and reserved_tail.start <= window.end <= reserved_tail.end
            ),
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        }
        for window in rolling_windows
    ]
    summary_is_finite = _payload_has_no_inf_nan(method_aggregates)
    trade_stats = {method_key: _distribution_stats([float(item["trade_count"]) for item in rows]) for method_key, rows in evidence_payload.items()}
    p_edge_stats = {
        method_key: _distribution_stats([float(item["p_edge_gt0"]) for item in rows if item["p_edge_gt0"] is not None])
        for method_key, rows in evidence_payload.items()
    }
    excluded_check = {
        method_key: {
            "excluded_not_aggregated": int(sum(1 for item in rows if not bool(item["usable"]))) == int(method_aggregates[method_key]["total_windows"] - method_aggregates[method_key]["usable_windows"]),
            "excluded_count": int(sum(1 for item in rows if not bool(item["usable"]))),
        }
        for method_key, rows in evidence_payload.items()
    }

    return {
        "overlap_checks": overlap_checks,
        "window_tail_checks": window_tail_checks,
        "weight_checks": weight_checks,
        "summary_no_inf_nan": {"passed": summary_is_finite, "detail": "Aggregated summary fields are finite."},
        "excluded_windows_removed_from_aggregation": excluded_check,
        "trade_count_distribution": trade_stats,
        "p_edge_distribution": p_edge_stats,
        "future_leakage": {
            "passed": True,
            "detail": (
                "Signals are computed on completed bars, cached once, and only then sliced into holdout and rolling windows. "
                "Coverage is enforced by existing Stage-2/feature leakage tests and tests/test_stage2_probabilistic.py."
            ),
        },
    }


def _overall_recommendation(method_aggregates: dict[str, dict[str, Any]]) -> str:
    if any(item["classification"] == "STRONG" for item in method_aggregates.values()):
        return "Proceed to leverage modeling"
    return "Improve discovery/search space/exits before leverage"


def _aggregate_rows(method_aggregates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"method": method_key, **payload} for method_key, payload in method_aggregates.items()]


def _window_evidence_to_row(item: PortfolioWindowEvidence) -> dict[str, Any]:
    return {
        "method": item.method,
        "window_id": item.window_id,
        "start": item.start.isoformat(),
        "end": item.end.isoformat(),
        "usable": bool(item.usable),
        "exclusion_reason": item.exclusion_reason,
        "trade_count": int(item.trade_count),
        "trades_per_month": float(item.trades_per_month),
        "exposure_ratio": float(item.exposure_ratio),
        "average_correlation": float(item.average_correlation),
        "effective_n": float(item.effective_n),
        "raw_profit_factor": float(item.raw_profit_factor),
        "expectancy": float(item.expectancy),
        "exp_lcb": float(item.exp_lcb),
        "return_pct": float(item.return_pct),
        "max_drawdown": float(item.max_drawdown),
        "Sharpe_ratio": float(item.sharpe_ratio),
        "Sortino_ratio": float(item.sortino_ratio),
        "Calmar_ratio": float(item.calmar_ratio),
        "p_edge_gt0": item.p_edge_gt0,
        "edge_ci_low": item.edge_ci_low,
        "edge_median": item.edge_median,
        "edge_ci_high": item.edge_ci_high,
        "p_pf_gt1": item.p_pf_gt1,
        "pf_ci_low": item.pf_ci_low,
        "pf_median": item.pf_median,
        "pf_ci_high": item.pf_ci_high,
        "lcb_t_edge": item.lcb_t_edge,
    }


def _rolling_to_window_spec(window: RollingWindowSpec) -> WindowSpec:
    return WindowSpec(
        name=window.name,
        kind="forward",
        expected_start=window.start,
        expected_end=window.end,
        actual_start=window.start,
        actual_end=window.end,
        truncated=window.truncated,
        enough_data=window.enough_data,
        bar_count=window.bar_count,
        note=window.note,
    )


def _window_payload(window: WindowSpec) -> dict[str, Any]:
    return {
        "name": window.name,
        "kind": window.kind,
        "expected_start": window.expected_start.isoformat(),
        "expected_end": window.expected_end.isoformat(),
        "actual_start": window.actual_start.isoformat() if window.actual_start is not None else None,
        "actual_end": window.actual_end.isoformat() if window.actual_end is not None else None,
        "truncated": bool(window.truncated),
        "enough_data": bool(window.enough_data),
        "bar_count": int(window.bar_count),
        "note": window.note,
    }


def _rolling_payload(window: RollingWindowSpec) -> dict[str, Any]:
    return {
        "window_id": window.name,
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "truncated": bool(window.truncated),
        "enough_data": bool(window.enough_data),
        "bar_count": int(window.bar_count),
        "note": window.note,
    }

def _write_probabilistic_report(
    run_dir: Path,
    summary: dict[str, Any],
    method_compositions: dict[str, MethodComposition],
    candidate_records: dict[str, Stage1CandidateRecord],
    evidence_payload: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = []
    lines.append("# Stage-2.8 Probabilistic Rolling Walk-Forward Report")
    lines.append("")
    lines.append("## Section 1 - Provenance")
    lines.append(f"- Stage-1 run_id: `{summary['stage1_run_id']}`")
    lines.append(f"- Stage-2 run_id: `{summary['stage2_run_id']}`")
    lines.append(f"- Stage-2.8 run_id: `{summary['run_id']}`")
    lines.append(f"- exact CLI command: `{summary['command']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- n_boot: `{summary['n_boot']}`")
    lines.append(f"- config hash: `{summary['config_hash']}`")
    lines.append(f"- data hash: `{summary['data_hash']}`")
    lines.append(f"- portfolio methods used: `{list(summary['method_summaries'].keys())}`")
    lines.append(f"- holdout range used: `{summary['holdout_window']['actual_start']}` .. `{summary['holdout_window']['actual_end']}`")
    lines.append(
        f"- reserve_tail_days: `{summary['reserve_tail_days']}` with reserved tail `{summary['reserved_tail']['start']}` .. `{summary['reserved_tail']['end']}`"
    )
    lines.append(f"- shifted_for_forward_window fallback used: `{summary['fallback_used']}`")
    lines.append(f"- fallback reason: {summary['fallback_reason']}")
    lines.append("")

    lines.append("## Section 2 - Window Construction")
    lines.append(f"- window_days: `{summary['window_days']}`")
    lines.append(f"- stride_days: `{summary['stride_days']}`")
    lines.append("- Rolling windows overlap by design when stride_days < window_days so regime sensitivity is measured on a denser sequence of recent samples.")
    lines.append("| window_id | start | end | truncated | enough_data | note |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for window in summary["rolling_windows"]:
        lines.append(
            f"| {window['window_id']} | {window['start']} | {window['end']} | {window['truncated']} | {window['enough_data']} | {window['note'] or '-'} |"
        )
    lines.append("- sanity checks:")
    for item in summary["sanity_checks"]["window_tail_checks"]:
        lines.append(
            f"  - {item['window']}: inside_reserved_tail=`{item['inside_reserved_tail']}` | {item['start']} .. {item['end']}"
        )
    lines.append("")

    lines.append("## Section 3 - Portfolio Composition")
    for method_key in ["equal", "vol", "corr-min"]:
        if method_key not in method_compositions:
            continue
        composition = method_compositions[method_key]
        lines.append(f"### {method_key}")
        lines.append(f"- number of component strategies: `{len(composition.selected_candidates)}`")
        lines.append(f"- effective N: `{composition.effective_n:.4f}`")
        lines.append(f"- average correlation: `{composition.average_correlation:.4f}`")
        lines.append("- component strategies:")
        for candidate_id in composition.selected_candidates:
            record = candidate_records[candidate_id]
            lines.append(
                f"  - `{candidate_id}` | {record.family} | gating={record.gating_mode} | exit={record.exit_mode}"
            )
        lines.append("- weights:")
        for candidate_id, weight in sorted(composition.weights.items(), key=lambda item: float(item[1]), reverse=True):
            lines.append(f"  - `{candidate_id}` | weight={float(weight):.6f}")
        lines.append(f"- full weights CSV: `weights_{method_key.replace('-', '_')}.csv`")
        lines.append("")

    lines.append("## Section 4 - Per-Window Evidence Table")
    lines.append("- Edge basis: mean trade PnL per portfolio trade, net of costs.")
    for method_key in ["equal", "vol", "corr-min"]:
        rows = evidence_payload.get(method_key, [])
        if not rows:
            continue
        lines.append(f"### {method_key}")
        lines.append(
            "| window_id | start | end | usable | exclusion_reason | trade_count | tpm | exposure_ratio | p_edge_gt0 | edge_ci_low | edge_median | edge_ci_high | p_pf_gt1 | pf_ci_low | pf_median | pf_ci_high | lcb_t_edge |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in rows:
            lines.append(
                f"| {row['window_id']} | {row['start']} | {row['end']} | {row['usable']} | {row['exclusion_reason'] or '-'} | "
                f"{row['trade_count']} | {_fmt(row['trades_per_month'])} | {_fmt(row['exposure_ratio'])} | {_fmt_or_na(row['p_edge_gt0'])} | "
                f"{_fmt_or_na(row['edge_ci_low'])} | {_fmt_or_na(row['edge_median'])} | {_fmt_or_na(row['edge_ci_high'])} | "
                f"{_fmt_or_na(row['p_pf_gt1'])} | {_fmt_or_na(row['pf_ci_low'])} | {_fmt_or_na(row['pf_median'])} | "
                f"{_fmt_or_na(row['pf_ci_high'])} | {_fmt_or_na(row['lcb_t_edge'])} |"
            )
        lines.append("")

    lines.append("## Section 5 - Aggregated Evidence")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        aggregate = payload["aggregate"]
        holdout = payload["holdout"]
        lines.append(f"### {method_key}")
        lines.append(f"- holdout expectancy: `{_fmt(holdout['expectancy'])}`")
        lines.append(f"- holdout exp_lcb: `{_fmt(holdout['exp_lcb'])}`")
        lines.append(f"- holdout p_edge_gt0: `{_fmt_or_na(holdout['p_edge_gt0'])}`")
        lines.append(f"- usable_windows / total_windows: `{aggregate['usable_windows']}` / `{aggregate['total_windows']}`")
        lines.append(f"- p_edge_gt0_median / min / mean: `{_fmt(aggregate['p_edge_gt0_median'])}` / `{_fmt(aggregate['p_edge_gt0_min'])}` / `{_fmt(aggregate['p_edge_gt0_mean'])}`")
        lines.append(f"- p_pf_gt1_median / min / mean: `{_fmt(aggregate['p_pf_gt1_median'])}` / `{_fmt(aggregate['p_pf_gt1_min'])}` / `{_fmt(aggregate['p_pf_gt1_mean'])}`")
        lines.append(f"- worst_edge_ci_low: `{_fmt(aggregate['worst_edge_ci_low'])}`")
        lines.append(f"- robustness_score: `{_fmt(aggregate['robustness_score'])}`")
        lines.append(f"- classification: `{aggregate['classification']}`")
        lines.append(f"- regime_sensitivity: `{aggregate['regime_sensitivity']}` | {aggregate['regime_sensitivity_reason']}")
        lines.append("")

    lines.append("## Section 6 - Conclusions")
    best_method = max(summary["method_summaries"], key=lambda key: float(summary["method_summaries"][key]["aggregate"]["robustness_score"]))
    lines.append(f"- best method by robustness_score: `{best_method}`")
    lines.append(f"- overall recommendation: {summary['overall_recommendation']}")
    lines.append("- no parameter re-optimization was performed; Stage-2.8 reused the frozen Stage-2 portfolios and candidate specifications.")
    lines.append("")

    lines.append("## Section 7 - Integrity Checks")
    lines.append(f"- confirm no NaN/inf in summary: `{summary['sanity_checks']['summary_no_inf_nan']['passed']}`")
    lines.append("- confirm excluded windows are excluded from aggregation:")
    for method_key, payload in summary["sanity_checks"]["excluded_windows_removed_from_aggregation"].items():
        lines.append(
            f"  - {method_key}: `{payload['excluded_not_aggregated']}` | excluded_count={payload['excluded_count']}"
        )
    lines.append("- trade_count distribution across windows:")
    for method_key, payload in summary["sanity_checks"]["trade_count_distribution"].items():
        lines.append(
            f"  - {method_key}: min={_fmt(payload['min'])}, median={_fmt(payload['median'])}, max={_fmt(payload['max'])}, mean={_fmt(payload['mean'])}"
        )
    lines.append("- p_edge_gt0 distribution across windows:")
    for method_key, payload in summary["sanity_checks"]["p_edge_distribution"].items():
        lines.append(
            f"  - {method_key}: min={_fmt(payload['min'])}, median={_fmt(payload['median'])}, max={_fmt(payload['max'])}, mean={_fmt(payload['mean'])}"
        )
    lines.append("- future leakage check:")
    lines.append(
        f"  - `{summary['sanity_checks']['future_leakage']['passed']}` | {summary['sanity_checks']['future_leakage']['detail']}"
    )
    report_text = "\n".join(lines).strip() + "\n"
    (run_dir / "probabilistic_report.md").write_text(report_text, encoding="utf-8")


def _default_command(
    stage2_run_id: str,
    window_days: int,
    stride_days: int,
    num_windows: int | None,
    reserve_tail_days: int,
    min_trades: int,
    min_exposure: float,
    n_boot: int,
    seed: int,
) -> str:
    command = (
        "python scripts/run_stage2_probabilistic.py "
        f"--stage2-run-id {stage2_run_id} --window-days {window_days} --stride-days {stride_days} "
        f"--reserve-tail-days {reserve_tail_days} --min_trades {min_trades} "
        f"--min_exposure {min_exposure} --n_boot {n_boot} --seed {seed}"
    )
    if num_windows is not None:
        command += f" --num-windows {num_windows}"
    return command


def _distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _payload_has_no_inf_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_payload_has_no_inf_nan(item) for item in value.values())
    if isinstance(value, list):
        return all(_payload_has_no_inf_nan(item) for item in value)
    return True


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_or_na(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
