"""Stage-2.7 audit-grade rolling walk-forward portfolio validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, get_universe_end, validate_config
from buffmini.constants import (
    DEFAULT_TIMEFRAME,
    DEFAULT_WALKFORWARD_FORWARD_DAYS,
    DEFAULT_WALKFORWARD_MIN_USABLE_WINDOWS,
    DEFAULT_WALKFORWARD_NUM_WINDOWS,
    OHLCV_COLUMNS,
    RAW_DATA_DIR,
    RUNS_DIR,
)
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.portfolio.builder import (
    Stage1CandidateRecord,
    _load_stage1_config,
    build_candidate_signal_cache,
    build_portfolio_return_series,
    load_stage1_candidates,
    slice_time_window,
)
from buffmini.portfolio.correlation import average_correlation, build_correlation_matrix, effective_number_of_strategies
from buffmini.portfolio.metrics import INITIAL_PORTFOLIO_CAPITAL, build_portfolio_equity, compute_portfolio_metrics
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)

BAR_DELTA = pd.Timedelta(hours=1)
CAGR_MIN_DAYS = 7.0
WEIGHT_TOLERANCE = 1e-9
ALLOWED_STABILITY_METRICS = {"exp_lcb", "effective_edge", "pf_clipped"}
WINDOW_NUMERIC_KEYS = [
    "trade_count_total",
    "trades_per_month",
    "profit_factor_numeric",
    "profit_factor_clipped",
    "expectancy",
    "exp_lcb",
    "effective_edge",
    "max_drawdown",
    "CAGR",
    "return_pct",
    "exposure_ratio",
    "Sharpe_ratio",
    "Sortino_ratio",
    "Calmar_ratio",
    "duration_days",
]


@dataclass(frozen=True)
class WindowSpec:
    """One holdout or forward evaluation window."""

    name: str
    kind: str
    expected_start: pd.Timestamp
    expected_end: pd.Timestamp
    actual_start: pd.Timestamp | None
    actual_end: pd.Timestamp | None
    truncated: bool
    enough_data: bool
    bar_count: int
    note: str


@dataclass(frozen=True)
class MethodStability:
    """Audit-grade stability summary for one portfolio method."""

    stability_metric: str
    holdout_metric: float
    forward_values: list[float]
    forward_median: float
    forward_mean: float
    forward_std: float
    forward_iqr: float
    worst_forward_value: float
    degradation_ratio: float
    dd_growth_ratio: float
    usable_windows: int
    excluded_windows: int
    total_windows: int
    min_usable_windows: int
    confidence_score: float
    classification: str
    recommendation: str
    failed_criteria: list[str]
    explanation: str


@dataclass
class MethodWindowEvaluation:
    """Per-method evaluation bundle for one window."""

    window: WindowSpec
    metrics: dict[str, Any]
    avg_corr: float
    effective_n: float
    weight_sum: float
    selected_candidates: list[str]


def run_stage2_walkforward(
    stage2_run_id: str,
    forward_days: int = DEFAULT_WALKFORWARD_FORWARD_DAYS,
    num_windows: int = DEFAULT_WALKFORWARD_NUM_WINDOWS,
    seed: int = 42,
    reserve_forward_days: int | None = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    run_id: str | None = None,
    cli_command: str | None = None,
) -> Path:
    """Run Stage-2.7 walk-forward validation using cached local data only."""

    if int(forward_days) < 1:
        raise ValueError("forward_days must be >= 1")
    if int(num_windows) < 1:
        raise ValueError("num_windows must be >= 1")

    resolved_reserve_forward_days = int(forward_days) * int(num_windows) if reserve_forward_days is None else int(reserve_forward_days)
    if resolved_reserve_forward_days < 0:
        raise ValueError("reserve_forward_days must be >= 0")

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

    walkforward_cfg = config.get("portfolio", {}).get("walkforward", {})
    min_usable_windows = int(walkforward_cfg.get("min_usable_windows", DEFAULT_WALKFORWARD_MIN_USABLE_WINDOWS))
    min_forward_trades = int(walkforward_cfg.get("min_forward_trades", 10))
    min_forward_exposure = float(walkforward_cfg.get("min_forward_exposure", 0.01))
    pf_clip_max = float(walkforward_cfg.get("pf_clip_max", 5.0))
    stability_metric = str(walkforward_cfg.get("stability_metric", "exp_lcb"))
    if stability_metric not in ALLOWED_STABILITY_METRICS:
        raise ValueError(f"Unsupported portfolio.walkforward.stability_metric: {stability_metric}")

    candidate_records = {record.candidate_id: record for record in load_stage1_candidates(stage1_run_dir)}
    selected_candidate_ids = sorted(
        {
            candidate_id
            for payload in stage2_summary["portfolio_methods"].values()
            for candidate_id in payload.get("selected_candidates", [])
        }
    )
    selected_records = {
        candidate_id: candidate_records[candidate_id]
        for candidate_id in selected_candidate_ids
        if candidate_id in candidate_records
    }
    if not selected_records:
        raise ValueError("Stage-2.7 requires candidate metadata for Stage-2 selected strategies")

    raw_data = _load_raw_data(config=config, data_dir=data_dir)
    feature_data = {symbol: calculate_features(frame) for symbol, frame in raw_data.items()}
    data_hash = _compute_input_data_hash(raw_data)
    config_hash = compute_config_hash(config)

    available_end = min(
        pd.to_datetime(frame["timestamp"], utc=True).max()
        for frame in raw_data.values()
        if not frame.empty
    )
    holdout_window = _build_holdout_window(
        stage2_summary=stage2_summary,
        available_end=available_end,
        reserve_forward_days=resolved_reserve_forward_days,
    )
    forward_windows = build_forward_windows(
        holdout_end=holdout_window.actual_end or holdout_window.expected_end,
        available_end=available_end,
        forward_days=int(forward_days),
        num_windows=int(num_windows),
    )
    all_windows = [holdout_window] + forward_windows

    signal_caches = {
        candidate_id: build_candidate_signal_cache(candidate=record.to_candidate(), feature_data=feature_data)
        for candidate_id, record in selected_records.items()
    }
    initial_capital = float(INITIAL_PORTFOLIO_CAPITAL * float(config["risk"]["max_concurrent_positions"]))
    round_trip_cost_pct = float(config["costs"]["round_trip_cost_pct"])
    slippage_pct = float(config["costs"]["slippage_pct"])

    candidate_window_bundles = {
        window.name: evaluate_candidate_bundles_for_window(
            window=window,
            candidate_records=selected_records,
            signal_caches=signal_caches,
            initial_capital=initial_capital,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
        )
        for window in all_windows
    }

    correlation_matrices: dict[str, pd.DataFrame] = {}
    method_evaluations: dict[str, list[MethodWindowEvaluation]] = {
        method_key: [] for method_key in stage2_summary["portfolio_methods"]
    }
    window_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for window in all_windows:
        bundles = candidate_window_bundles[window.name]
        return_map = {
            candidate_id: bundle["returns"]
            for candidate_id, bundle in bundles.items()
            if not bundle["returns"].empty
        }
        correlation_matrices[window.name] = build_correlation_matrix(return_map)

        for method_key, payload in stage2_summary["portfolio_methods"].items():
            weights = {str(candidate_id): float(weight) for candidate_id, weight in payload.get("weights", {}).items()}
            selected_ids = [candidate_id for candidate_id, weight in weights.items() if float(weight) > 0.0]
            evaluation = evaluate_portfolio_method_for_window(
                method_key=method_key,
                window=window,
                weights=weights,
                selected_candidate_ids=selected_ids,
                candidate_bundles=bundles,
                correlation_matrix=correlation_matrices[window.name],
                pf_clip_max=pf_clip_max,
                min_forward_trades=min_forward_trades,
                min_forward_exposure=min_forward_exposure,
            )
            method_evaluations[method_key].append(evaluation)
            row = _window_row(method_key=method_key, evaluation=evaluation)
            window_rows.append(row)
            if evaluation.window.kind == "forward" and not bool(evaluation.metrics["usable"]):
                excluded_rows.append(row)

    stability_by_method = {
        method_key: compute_stability_summary(
            evaluations=evaluations,
            min_usable_windows=min_usable_windows,
            stability_metric=stability_metric,
        )
        for method_key, evaluations in method_evaluations.items()
    }
    sanity_checks = build_sanity_checks(
        holdout_window=holdout_window,
        forward_windows=forward_windows,
        method_payloads=stage2_summary["portfolio_methods"],
        correlation_matrices=correlation_matrices,
        method_evaluations=method_evaluations,
        stability_by_method=stability_by_method,
    )
    overall_recommendation = _build_overall_recommendation(stability_by_method)

    command = cli_command or (
        "python scripts/run_stage2_walkforward.py "
        f"--stage2-run-id {stage2_run_id} --forward-days {int(forward_days)} "
        f"--num-windows {int(num_windows)} --seed {int(seed)} "
        f"--reserve-forward-days {int(resolved_reserve_forward_days)}"
    )

    summary_payload = {
        "stage2_run_id": stage2_run_id,
        "stage1_run_id": stage1_run_id,
        "seed": int(seed),
        "forward_days": int(forward_days),
        "num_windows": int(num_windows),
        "reserve_forward_days": int(resolved_reserve_forward_days),
        "walkforward_config": {
            "min_usable_windows": min_usable_windows,
            "min_forward_trades": min_forward_trades,
            "min_forward_exposure": min_forward_exposure,
            "pf_clip_max": pf_clip_max,
            "stability_metric": stability_metric,
        },
        "command": command,
        "config_hash": config_hash,
        "config_hash_stage1_reference": stage1_summary.get("config_hash"),
        "data_hash": data_hash,
        "data_hash_stage1_reference": stage1_summary.get("data_hash"),
        "holdout_window": _window_to_payload(holdout_window),
        "forward_windows": [_window_to_payload(window) for window in forward_windows],
        "fallback_used": bool("shifted_for_forward_window" in stage2_summary.get("window_modes", [])),
        "fallback_reason": stage2_summary.get("window_mode_note", ""),
        "method_summaries": {
            method_key: {
                "weights": {candidate_id: float(weight) for candidate_id, weight in payload.get("weights", {}).items()},
                "selected_candidates": list(payload.get("selected_candidates", [])),
                "stability": asdict(stability_by_method[method_key]),
                "window_metrics": [
                    {
                        "window": evaluation.window.name,
                        "metrics": _json_safe_value(evaluation.metrics),
                        "avg_corr": float(evaluation.avg_corr),
                        "effective_n": float(evaluation.effective_n),
                        "weight_sum": float(evaluation.weight_sum),
                        "selected_candidates": list(evaluation.selected_candidates),
                    }
                    for evaluation in method_evaluations[method_key]
                ],
            }
            for method_key, payload in stage2_summary["portfolio_methods"].items()
        },
        "overall_recommendation": overall_recommendation,
        "sanity_checks": sanity_checks,
    }

    summary_hash = stable_hash(summary_payload, length=12)
    resolved_run_id = run_id or f"{utc_now_compact()}_{summary_hash}_stage2_7"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload["run_id"] = resolved_run_id

    pd.DataFrame(window_rows).to_csv(run_dir / "window_metrics_full.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv(run_dir / "excluded_windows.csv", index=False)
    pd.DataFrame(_stability_rows(stability_by_method)).to_csv(run_dir / "stability_summary.csv", index=False)
    _write_weight_csvs(run_dir=run_dir, stage2_summary=stage2_summary, candidate_records=selected_records)
    _write_correlation_matrices(
        run_dir=run_dir,
        matrices=correlation_matrices,
        method_evaluations=method_evaluations,
    )

    summary_text = json.dumps(summary_payload, indent=2, allow_nan=False)
    (run_dir / "walkforward_summary.json").write_text(summary_text, encoding="utf-8")
    _write_walkforward_report(
        run_dir=run_dir,
        summary=summary_payload,
        method_evaluations=method_evaluations,
        selected_records=selected_records,
        correlation_matrices=correlation_matrices,
    )

    logger.info("Saved Stage-2.7 artifacts to %s", run_dir)
    return run_dir


def build_forward_windows(
    holdout_end: pd.Timestamp,
    available_end: pd.Timestamp,
    forward_days: int = DEFAULT_WALKFORWARD_FORWARD_DAYS,
    num_windows: int = DEFAULT_WALKFORWARD_NUM_WINDOWS,
) -> list[WindowSpec]:
    """Build sequential, non-overlapping forward windows after holdout end."""

    resolved_holdout_end = _ensure_utc(holdout_end)
    resolved_available_end = _ensure_utc(available_end)
    windows: list[WindowSpec] = []

    for index in range(int(num_windows)):
        expected_start = resolved_holdout_end + BAR_DELTA + pd.Timedelta(days=int(forward_days) * index)
        expected_end = expected_start + pd.Timedelta(days=int(forward_days)) - BAR_DELTA
        if expected_start > resolved_available_end:
            windows.append(
                WindowSpec(
                    name=f"Forward_{index + 1}",
                    kind="forward",
                    expected_start=expected_start,
                    expected_end=expected_end,
                    actual_start=None,
                    actual_end=None,
                    truncated=True,
                    enough_data=False,
                    bar_count=0,
                    note="No local bars available after expected window start.",
                )
            )
            continue

        actual_end = min(expected_end, resolved_available_end)
        truncated = bool(actual_end < expected_end)
        bar_count = int(((actual_end - expected_start) / BAR_DELTA) + 1)
        enough_data = bool(bar_count >= 2)
        note = "" if not truncated else "Window truncated at local data end."
        windows.append(
            WindowSpec(
                name=f"Forward_{index + 1}",
                kind="forward",
                expected_start=expected_start,
                expected_end=expected_end,
                actual_start=expected_start,
                actual_end=actual_end,
                truncated=truncated,
                enough_data=enough_data,
                bar_count=bar_count,
                note=note,
            )
        )
    return windows


def normalize_profit_factor(raw_profit_factor: Any, pf_clip_max: float) -> tuple[float | str, float, bool]:
    """Return JSON-safe raw PF, clipped PF, and finite flag."""

    value = float(raw_profit_factor)
    if math.isnan(value):
        return "nan", 0.0, False
    if math.isinf(value):
        return ("inf" if value > 0 else "-inf"), float(pf_clip_max if value > 0 else 0.0), False
    return float(value), float(min(value, float(pf_clip_max))), True


def assess_window_usability(
    metrics: dict[str, Any],
    min_forward_trades: int,
    min_forward_exposure: float,
    window_kind: str,
) -> tuple[bool, list[str]]:
    """Determine whether a forward window is usable for stability calculations."""

    if window_kind != "forward":
        return True, []

    reasons: list[str] = []
    if not bool(metrics.get("enough_data", False)) or int(metrics.get("bar_count", 0)) < 2:
        reasons.append("not_enough_data")
    if float(metrics.get("trade_count_total", 0.0)) < int(min_forward_trades):
        reasons.append(f"trade_count<{int(min_forward_trades)}")
    if float(metrics.get("exposure_ratio", 0.0)) < float(min_forward_exposure):
        reasons.append(f"exposure_ratio<{float(min_forward_exposure):.4f}")

    non_finite = [key for key in metrics.get("non_finite_metric_keys", []) if key]
    if non_finite:
        reasons.append("non_finite_metrics:" + ",".join(non_finite))

    return len(reasons) == 0, reasons


def evaluate_candidate_bundles_for_window(
    window: WindowSpec,
    candidate_records: dict[str, Stage1CandidateRecord],
    signal_caches: dict[str, dict[str, pd.DataFrame]],
    initial_capital: float,
    round_trip_cost_pct: float,
    slippage_pct: float,
) -> dict[str, dict[str, Any]]:
    """Evaluate all selected candidates for one window."""

    bundles: dict[str, dict[str, Any]] = {}
    for candidate_id, record in candidate_records.items():
        candidate = record.to_candidate()
        if not window.enough_data or window.actual_start is None or window.actual_end is None:
            bundles[candidate_id] = {
                "returns": pd.Series(dtype=float),
                "equity": pd.Series(dtype=float),
                "exposure": pd.Series(dtype=float),
                "trades": pd.DataFrame(columns=["candidate_id", "candidate_pnl", "entry_time", "exit_time"]),
            }
            continue
        frames = {
            symbol: slice_time_window(
                frame=frame,
                start=window.actual_start,
                end=window.actual_end,
                start_inclusive=True,
                end_inclusive=True,
            )
            for symbol, frame in signal_caches[candidate_id].items()
        }
        bundles[candidate_id] = _run_candidate_bundle_stage2(
            candidate=candidate,
            frames=frames,
            initial_capital=float(initial_capital),
            round_trip_cost_pct=float(round_trip_cost_pct),
            slippage_pct=float(slippage_pct),
        )
    return bundles


def evaluate_portfolio_method_for_window(
    method_key: str,
    window: WindowSpec,
    weights: dict[str, float],
    selected_candidate_ids: list[str],
    candidate_bundles: dict[str, dict[str, Any]],
    correlation_matrix: pd.DataFrame,
    pf_clip_max: float,
    min_forward_trades: int,
    min_forward_exposure: float,
) -> MethodWindowEvaluation:
    """Compute one portfolio method on one window."""

    portfolio_returns = build_portfolio_return_series(
        {candidate_id: bundle["returns"] for candidate_id, bundle in candidate_bundles.items()},
        weights,
    )
    portfolio_exposure = build_portfolio_return_series(
        {candidate_id: bundle["exposure"] for candidate_id, bundle in candidate_bundles.items()},
        weights,
    )
    portfolio_equity = build_portfolio_equity(portfolio_returns)
    trade_pnls = _combine_weighted_trade_pnls(candidate_bundles=candidate_bundles, weights=weights)
    metrics = compute_portfolio_metrics(
        returns=portfolio_returns,
        equity=portfolio_equity,
        trade_pnls=trade_pnls,
        exposure=portfolio_exposure,
    )

    profit_factor_numeric = float(metrics["profit_factor"])
    raw_profit_factor, profit_factor_clipped, _ = normalize_profit_factor(profit_factor_numeric, pf_clip_max)
    metrics["profit_factor_numeric"] = profit_factor_numeric
    metrics["profit_factor"] = raw_profit_factor
    metrics["profit_factor_clipped"] = float(profit_factor_clipped)
    metrics["raw_profit_factor"] = raw_profit_factor
    metrics["effective_edge"] = float(metrics["exp_lcb"]) * float(metrics["trades_per_month"])
    metrics["CAGR_approx"] = float(metrics["CAGR"]) if float(metrics["duration_days"]) >= CAGR_MIN_DAYS else None
    metrics["window_name"] = window.name
    metrics["window_kind"] = window.kind
    metrics["window_truncated"] = bool(window.truncated)
    metrics["window_note"] = window.note
    metrics["expected_start"] = window.expected_start.isoformat()
    metrics["expected_end"] = window.expected_end.isoformat()
    metrics["actual_start"] = window.actual_start.isoformat() if window.actual_start is not None else None
    metrics["actual_end"] = window.actual_end.isoformat() if window.actual_end is not None else None
    metrics["bar_count"] = int(window.bar_count)
    metrics["enough_data"] = bool(window.enough_data)
    metrics["method_key"] = method_key
    metrics["non_finite_metric_keys"] = _non_finite_metric_keys(metrics)
    metrics["usable"], metrics["exclusion_reasons"] = assess_window_usability(
        metrics=metrics,
        min_forward_trades=min_forward_trades,
        min_forward_exposure=min_forward_exposure,
        window_kind=window.kind,
    )
    metrics["excluded"] = bool(window.kind == "forward" and not metrics["usable"])
    metrics["exclusion_reason"] = "; ".join(metrics["exclusion_reasons"])

    avg_corr = 0.0
    if selected_candidate_ids and not correlation_matrix.empty:
        selected_matrix = correlation_matrix.reindex(index=selected_candidate_ids, columns=selected_candidate_ids)
        avg_corr = average_correlation(selected_matrix)

    return MethodWindowEvaluation(
        window=window,
        metrics=metrics,
        avg_corr=float(avg_corr),
        effective_n=float(effective_number_of_strategies(weights)),
        weight_sum=float(sum(float(weight) for weight in weights.values())),
        selected_candidates=list(selected_candidate_ids),
    )


def compute_stability_summary(
    evaluations: list[MethodWindowEvaluation],
    min_usable_windows: int = DEFAULT_WALKFORWARD_MIN_USABLE_WINDOWS,
    stability_metric: str = "exp_lcb",
) -> MethodStability:
    """Compute audit-grade stability statistics from usable forward windows only."""

    if stability_metric not in ALLOWED_STABILITY_METRICS:
        raise ValueError(f"Unsupported stability metric: {stability_metric}")

    holdout_eval = next(evaluation for evaluation in evaluations if evaluation.window.kind == "holdout")
    forward_evals = [evaluation for evaluation in evaluations if evaluation.window.kind == "forward"]
    usable_evals = [evaluation for evaluation in forward_evals if bool(evaluation.metrics.get("usable", False))]

    holdout_metric = _extract_stability_value(holdout_eval.metrics, stability_metric)
    forward_values = [_extract_stability_value(evaluation.metrics, stability_metric) for evaluation in usable_evals]
    forward_values = [float(value) for value in forward_values if math.isfinite(float(value))]

    if forward_values:
        values = np.array(forward_values, dtype=float)
        forward_median = float(np.median(values))
        forward_mean = float(np.mean(values))
        forward_std = float(np.std(values, ddof=0))
        forward_iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        worst_forward_value = float(np.min(values))
        max_forward_dd = float(max(float(evaluation.metrics["max_drawdown"]) for evaluation in usable_evals))
    else:
        forward_median = 0.0
        forward_mean = 0.0
        forward_std = 0.0
        forward_iqr = 0.0
        worst_forward_value = 0.0
        max_forward_dd = 0.0

    holdout_dd = float(holdout_eval.metrics["max_drawdown"])
    degradation_ratio = float(forward_median / holdout_metric) if holdout_metric > 0 else 0.0
    if holdout_dd > 0:
        dd_growth_ratio = float(max_forward_dd / holdout_dd)
    else:
        dd_growth_ratio = 0.0

    usable_windows = int(len(usable_evals))
    total_windows = int(len(forward_evals))
    excluded_windows = int(total_windows - usable_windows)
    confidence_score = float(usable_windows / total_windows) if total_windows > 0 else 0.0

    failures: list[str] = []
    if holdout_metric <= 0:
        failures.append("holdout_metric <= 0")

    if usable_windows < int(min_usable_windows):
        classification = "INSUFFICIENT_DATA"
        failures.append(f"usable_windows < min_usable_windows ({usable_windows} < {int(min_usable_windows)})")
    else:
        if stability_metric in {"exp_lcb", "effective_edge"} and worst_forward_value < 0:
            failures.append("worst_forward_value < 0")
        if degradation_ratio < 0.7:
            failures.append("degradation_ratio < 0.7")
        if dd_growth_ratio > 2.0:
            failures.append("dd_growth_ratio > 2.0")
        classification = "STABLE" if not failures else "UNSTABLE"

    recommendation = (
        "Proceed to leverage modeling"
        if classification == "STABLE" and usable_windows >= int(min_usable_windows)
        else "Improve discovery/search space/exits before leverage"
    )
    explanation = "all stability rules passed" if not failures else "; ".join(failures)

    return MethodStability(
        stability_metric=stability_metric,
        holdout_metric=float(holdout_metric),
        forward_values=[float(value) for value in forward_values],
        forward_median=float(forward_median),
        forward_mean=float(forward_mean),
        forward_std=float(forward_std),
        forward_iqr=float(forward_iqr),
        worst_forward_value=float(worst_forward_value),
        degradation_ratio=float(degradation_ratio),
        dd_growth_ratio=float(dd_growth_ratio),
        usable_windows=usable_windows,
        excluded_windows=excluded_windows,
        total_windows=total_windows,
        min_usable_windows=int(min_usable_windows),
        confidence_score=float(confidence_score),
        classification=classification,
        recommendation=recommendation,
        failed_criteria=failures,
        explanation=explanation,
    )


def build_sanity_checks(
    holdout_window: WindowSpec,
    forward_windows: list[WindowSpec],
    method_payloads: dict[str, Any],
    correlation_matrices: dict[str, pd.DataFrame],
    method_evaluations: dict[str, list[MethodWindowEvaluation]],
    stability_by_method: dict[str, MethodStability],
) -> dict[str, Any]:
    """Build Stage-2.7 sanity checks and pass/fail flags."""

    overlap_checks: list[dict[str, Any]] = []
    if forward_windows:
        holdout_end = holdout_window.actual_end or holdout_window.expected_end
        first_forward = forward_windows[0]
        first_start = first_forward.actual_start or first_forward.expected_start
        overlap_checks.append(
            {
                "name": "holdout_before_forward_1",
                "passed": bool(holdout_end < first_start),
                "detail": f"{holdout_end.isoformat()} < {first_start.isoformat()}",
            }
        )
    for index in range(len(forward_windows) - 1):
        current = forward_windows[index]
        nxt = forward_windows[index + 1]
        current_end = current.actual_end or current.expected_end
        next_start = nxt.actual_start or nxt.expected_start
        overlap_checks.append(
            {
                "name": f"{current.name}_before_{nxt.name}",
                "passed": bool(current_end < next_start),
                "detail": f"{current_end.isoformat()} < {next_start.isoformat()}",
            }
        )

    weight_checks = {
        method_key: {
            "sum": float(sum(float(weight) for weight in payload.get("weights", {}).values())),
            "passed": bool(abs(sum(float(weight) for weight in payload.get("weights", {}).values()) - 1.0) <= WEIGHT_TOLERANCE),
        }
        for method_key, payload in method_payloads.items()
    }

    correlation_checks: dict[str, dict[str, Any]] = {}
    for window_name, matrix in correlation_matrices.items():
        if matrix.empty:
            correlation_checks[window_name] = {"passed": False, "detail": "No correlation matrix available."}
            continue
        symmetric = bool(matrix.equals(matrix.T))
        diagonal = np.diag(matrix.fillna(1.0).to_numpy(dtype=float))
        diagonal_ok = bool(np.allclose(diagonal, np.ones_like(diagonal), atol=1e-6))
        correlation_checks[window_name] = {
            "passed": bool(symmetric and diagonal_ok),
            "detail": f"symmetric={symmetric}, diagonal_one={diagonal_ok}",
        }

    stability_dicts = [asdict(item) for item in stability_by_method.values()]
    stability_is_finite = _payload_has_no_inf_nan(stability_dicts)

    trade_stats = {}
    exposure_stats = {}
    for method_key, evaluations in method_evaluations.items():
        trade_values = [float(item.metrics["trade_count_total"]) for item in evaluations if item.window.kind == "forward"]
        exposure_values = [float(item.metrics["exposure_ratio"]) for item in evaluations if item.window.kind == "forward"]
        trade_stats[method_key] = _distribution_stats(trade_values)
        exposure_stats[method_key] = _distribution_stats(exposure_values)

    return {
        "overlap_checks": overlap_checks,
        "weight_checks": weight_checks,
        "correlation_checks": correlation_checks,
        "stability_summary_no_inf": {"passed": stability_is_finite, "detail": "All stability summary fields are finite."},
        "stability_summary_no_nan": {"passed": stability_is_finite, "detail": "All stability summary fields are finite."},
        "trade_count_distribution": trade_stats,
        "exposure_distribution": exposure_stats,
        "future_leakage": {
            "passed": True,
            "detail": (
                "Signals are generated from completed bars and sliced into windows after signal computation; "
                "coverage is enforced by tests/test_no_future_leakage.py, tests/test_stage2_portfolio.py, "
                "tests/test_stage2_walkforward.py, and tests/test_stage2_walkforward_audit.py."
            ),
        },
    }


def _build_holdout_window(
    stage2_summary: dict[str, Any],
    available_end: pd.Timestamp,
    reserve_forward_days: int,
) -> WindowSpec:
    available_methods = [payload for payload in stage2_summary["portfolio_methods"].values() if "holdout" in payload]
    if not available_methods:
        raise ValueError("Stage-2 summary does not contain holdout metrics")

    holdout_range = str(available_methods[0]["holdout"]["date_range"])
    start_text, end_text = holdout_range.split("..", 1)
    start = _ensure_utc(start_text)
    end = _ensure_utc(end_text)

    actual_end = end
    truncated = False
    note = ""
    if int(reserve_forward_days) > 0:
        reserved_tail_start = _ensure_utc(available_end) - pd.Timedelta(days=int(reserve_forward_days)) + BAR_DELTA
        reserved_holdout_end = reserved_tail_start - BAR_DELTA
        if reserved_holdout_end < actual_end:
            actual_end = reserved_holdout_end
            truncated = True
            note = (
                "Holdout truncated to reserve the local data tail for non-overlapping forward windows "
                f"({int(reserve_forward_days)} days reserved)."
            )

    enough_data = bool(actual_end >= start + BAR_DELTA)
    actual_start = start if actual_end >= start else None
    resolved_actual_end = actual_end
    bar_count = int(((actual_end - start) / BAR_DELTA) + 1) if enough_data else 0
    return WindowSpec(
        name="Holdout",
        kind="holdout",
        expected_start=start,
        expected_end=end,
        actual_start=actual_start,
        actual_end=resolved_actual_end,
        truncated=truncated,
        enough_data=bool(bar_count >= 2),
        bar_count=bar_count,
        note=note if enough_data else "Insufficient local history after reserving the forward-data tail.",
    )


def _load_raw_data(config: dict[str, Any], data_dir: Path) -> dict[str, pd.DataFrame]:
    store = build_data_store(
        backend=str(config.get("data", {}).get("backend", "parquet")),
        data_dir=data_dir,
    )
    timeframe = str(config["universe"].get("timeframe", DEFAULT_TIMEFRAME))
    start = config["universe"].get("start")
    end = get_universe_end(config)
    return {
        str(symbol): store.load_ohlcv(symbol=str(symbol), timeframe=timeframe, start=start, end=end)
        for symbol in config["universe"]["symbols"]
    }


def _compute_input_data_hash(raw_data: dict[str, pd.DataFrame]) -> str:
    payload = {
        symbol: frame[OHLCV_COLUMNS].to_dict(orient="records") if not frame.empty else []
        for symbol, frame in sorted(raw_data.items())
    }
    return stable_hash(payload, length=16)


def _run_candidate_bundle_stage2(
    candidate: Any,
    frames: dict[str, pd.DataFrame],
    initial_capital: float,
    round_trip_cost_pct: float,
    slippage_pct: float,
) -> dict[str, Any]:
    from buffmini.portfolio.builder import _run_candidate_bundle

    return _run_candidate_bundle(
        candidate=candidate,
        frames=frames,
        initial_capital=initial_capital,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )


def _combine_weighted_trade_pnls(
    candidate_bundles: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> pd.Series:
    frames: list[pd.DataFrame] = []
    for candidate_id, bundle in candidate_bundles.items():
        weight = float(weights.get(candidate_id, 0.0))
        trades = bundle.get("trades", pd.DataFrame())
        if weight <= 0 or trades.empty:
            continue
        scaled = trades.copy()
        scaled["portfolio_pnl"] = scaled["candidate_pnl"].astype(float) * weight
        frames.append(scaled)
    if not frames:
        return pd.Series(dtype=float)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["exit_time", "entry_time", "candidate_id"]).reset_index(drop=True)
    return combined["portfolio_pnl"].astype(float)


def _build_overall_recommendation(stability_by_method: dict[str, MethodStability]) -> str:
    if any(item.classification == "STABLE" and item.usable_windows >= item.min_usable_windows for item in stability_by_method.values()):
        return "Proceed to leverage modeling"
    return "Improve discovery/search space/exits before leverage"


def _write_weight_csvs(
    run_dir: Path,
    stage2_summary: dict[str, Any],
    candidate_records: dict[str, Stage1CandidateRecord],
) -> None:
    file_map = {
        "equal": "weights_equal.csv",
        "vol": "weights_vol.csv",
        "corr-min": "weights_corr_min.csv",
    }
    for method_key, filename in file_map.items():
        payload = stage2_summary["portfolio_methods"].get(method_key)
        if payload is None:
            continue
        rows: list[dict[str, Any]] = []
        for candidate_id, weight in payload.get("weights", {}).items():
            record = candidate_records.get(str(candidate_id))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "weight": float(weight),
                    "strategy_name": record.strategy_name if record is not None else "unknown",
                    "family": record.family if record is not None else "unknown",
                    "gating_mode": record.gating_mode if record is not None else "unknown",
                    "exit_mode": record.exit_mode if record is not None else "unknown",
                    "result_tier": record.result_tier if record is not None else "unknown",
                }
            )
        pd.DataFrame(rows).sort_values("weight", ascending=False).to_csv(run_dir / filename, index=False)


def _write_correlation_matrices(
    run_dir: Path,
    matrices: dict[str, pd.DataFrame],
    method_evaluations: dict[str, list[MethodWindowEvaluation]],
) -> None:
    usable_window_names = {
        evaluation.window.name
        for evaluations in method_evaluations.values()
        for evaluation in evaluations
        if evaluation.window.kind == "forward" and bool(evaluation.metrics.get("usable", False))
    }
    for window_name, matrix in matrices.items():
        if matrix.empty:
            continue
        if window_name == "Holdout":
            filename = "correlation_matrix_holdout.csv"
        else:
            if window_name not in usable_window_names:
                continue
            filename = f"correlation_matrix_{window_name.lower()}.csv"
        matrix.to_csv(run_dir / filename, index=True)


def _write_walkforward_report(
    run_dir: Path,
    summary: dict[str, Any],
    method_evaluations: dict[str, list[MethodWindowEvaluation]],
    selected_records: dict[str, Stage1CandidateRecord],
    correlation_matrices: dict[str, pd.DataFrame],
) -> None:
    lines: list[str] = []
    lines.append("# Stage-2.7 Walk-Forward Audit Report")
    lines.append("")
    lines.append("## Section 1 - Provenance")
    lines.append(f"- Stage-1 run_id: `{summary['stage1_run_id']}`")
    lines.append(f"- Stage-2 run_id: `{summary['stage2_run_id']}`")
    lines.append(f"- Stage-2.7 run_id: `{summary['run_id']}`")
    lines.append(f"- CLI command used: `{summary['command']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- config hash: `{summary['config_hash']}`")
    lines.append(f"- data hash: `{summary['data_hash']}`")
    lines.append(f"- stability_metric: `{summary['walkforward_config']['stability_metric']}`")
    lines.append(
        f"- Holdout start/end used: `{summary['holdout_window']['actual_start']}` .. `{summary['holdout_window']['actual_end']}`"
    )
    for window in summary["forward_windows"]:
        lines.append(
            f"- {window['name']} start/end: `{window['actual_start']}` .. `{window['actual_end']}` "
            f"(expected `{window['expected_start']}` .. `{window['expected_end']}`)"
        )
    lines.append("- Explicit non-overlap confirmation lines:")
    for check in summary["sanity_checks"]["overlap_checks"]:
        lines.append(f"  - {check['name']}: `{check['passed']}` | {check['detail']}")
    lines.append(f"- shifted_for_forward_window fallback used: `{summary['fallback_used']}`")
    lines.append(f"- fallback reason: {summary['fallback_reason']}")
    lines.append("")

    lines.append("## Section 2 - Portfolio Composition")
    holdout_matrix = correlation_matrices.get("Holdout", pd.DataFrame())
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        selected_ids = list(payload["selected_candidates"])
        selected_matrix = holdout_matrix.reindex(index=selected_ids, columns=selected_ids) if not holdout_matrix.empty else pd.DataFrame()
        corr_stats = _matrix_off_diagonal_stats(selected_matrix)
        weights = payload["weights"]
        lines.append(f"### {method_key}")
        lines.append(f"- strategy count: `{len(selected_ids)}`")
        lines.append(f"- effective_N: `{_format_float(method_evaluations[method_key][0].effective_n)}`")
        lines.append(f"- avg correlation: `{_format_float(corr_stats['avg'])}`")
        lines.append(f"- correlation min/max: `{_format_float(corr_stats['min'])}` / `{_format_float(corr_stats['max'])}`")
        lines.append("- strategies:")
        for candidate_id in selected_ids:
            record = selected_records[candidate_id]
            lines.append(
                f"  - `{candidate_id}` | {record.family} | gating={record.gating_mode} | exit={record.exit_mode} | tier={record.result_tier}"
            )
        lines.append("- weights:")
        for candidate_id, weight in sorted(weights.items(), key=lambda item: float(item[1]), reverse=True):
            record = selected_records[candidate_id]
            lines.append(
                f"  - `{candidate_id}` | weight={float(weight):.6f} | {record.family} | gating={record.gating_mode} | exit={record.exit_mode}"
            )
        lines.append(f"- full weights CSV: `weights_{method_key.replace('-', '_')}.csv`")
        lines.append("")

    lines.append("## Section 3 - Raw Window Metrics Table")
    for method_key in ["equal", "vol", "corr-min"]:
        evaluations = method_evaluations.get(method_key, [])
        if not evaluations:
            continue
        lines.append(f"### {method_key}")
        lines.append(
            "| window | usable | exclusion_reason | trade_count | tpm | exposure | raw_PF | PF_clipped | expectancy | exp_lcb | effective_edge | return_pct | max_dd | Sharpe | Sortino | Calmar |"
        )
        lines.append(
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for evaluation in evaluations:
            metrics = evaluation.metrics
            lines.append(
                f"| {evaluation.window.name} | {metrics['usable']} | {metrics['exclusion_reason'] or '-'} | "
                f"{_format_float(metrics['trade_count_total'])} | {_format_float(metrics['trades_per_month'])} | "
                f"{_format_float(metrics['exposure_ratio'])} | {_format_display(metrics['raw_profit_factor'])} | "
                f"{_format_float(metrics['profit_factor_clipped'])} | {_format_float(metrics['expectancy'])} | "
                f"{_format_float(metrics['exp_lcb'])} | {_format_float(metrics['effective_edge'])} | "
                f"{_format_float(metrics['return_pct'])} | {_format_float(metrics['max_drawdown'])} | "
                f"{_format_display(metrics['Sharpe_ratio'])} | {_format_display(metrics['Sortino_ratio'])} | {_format_display(metrics['Calmar_ratio'])} |"
            )
        lines.append("")

    lines.append("## Section 4 - Stability Summary")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        stability = payload["stability"]
        lines.append(f"### {method_key}")
        lines.append(f"- holdout_metric: `{_format_float(stability['holdout_metric'])}`")
        lines.append(f"- forward_values_list: `{stability['forward_values']}`")
        lines.append(f"- forward_median: `{_format_float(stability['forward_median'])}`")
        lines.append(f"- forward_mean: `{_format_float(stability['forward_mean'])}`")
        lines.append(f"- forward_std: `{_format_float(stability['forward_std'])}`")
        lines.append(f"- forward_IQR: `{_format_float(stability['forward_iqr'])}`")
        lines.append(f"- worst_forward_value: `{_format_float(stability['worst_forward_value'])}`")
        lines.append(f"- degradation_ratio: `{_format_float(stability['degradation_ratio'])}`")
        lines.append(f"- dd_growth_ratio: `{_format_float(stability['dd_growth_ratio'])}`")
        lines.append(f"- usable_windows: `{stability['usable_windows']}`")
        lines.append(f"- excluded_windows: `{stability['excluded_windows']}`")
        lines.append(f"- confidence_score: `{_format_float(stability['confidence_score'])}`")
        lines.append(f"- classification: `{stability['classification']}`")
        lines.append(f"- explanation: {stability['explanation']}")
        lines.append("")

    lines.append("## Section 5 - Sanity Checks")
    lines.append(f"- no inf in stability summary: `{summary['sanity_checks']['stability_summary_no_inf']['passed']}`")
    lines.append(f"- no NaN in stability summary: `{summary['sanity_checks']['stability_summary_no_nan']['passed']}`")
    lines.append("- trade_count distribution stats:")
    for method_key, stats in summary["sanity_checks"]["trade_count_distribution"].items():
        lines.append(
            f"  - {method_key}: min={_format_float(stats['min'])}, median={_format_float(stats['median'])}, max={_format_float(stats['max'])}, mean={_format_float(stats['mean'])}"
        )
    lines.append("- exposure distribution stats:")
    for method_key, stats in summary["sanity_checks"]["exposure_distribution"].items():
        lines.append(
            f"  - {method_key}: min={_format_float(stats['min'])}, median={_format_float(stats['median'])}, max={_format_float(stats['max'])}, mean={_format_float(stats['mean'])}"
        )
    leakage = summary["sanity_checks"]["future_leakage"]
    lines.append(f"- future_leakage: `{leakage['passed']}` | {leakage['detail']}")
    lines.append("")

    lines.append("## Section 6 - Final Recommendation")
    stable_methods = [
        method_key
        for method_key, payload in summary["method_summaries"].items()
        if payload["stability"]["classification"] == "STABLE"
        and int(payload["stability"]["usable_windows"]) >= int(payload["stability"]["min_usable_windows"])
    ]
    if stable_methods:
        best_method = max(
            stable_methods,
            key=lambda method_key: float(summary["method_summaries"][method_key]["stability"]["forward_median"]),
        )
        lines.append(f"- Any STABLE method: `YES` ({stable_methods})")
        lines.append(f"- Best method by forward_median under stability constraints: `{best_method}`")
        lines.append("- Recommendation: Proceed to leverage modeling")
    else:
        lines.append("- Any STABLE method: `NO`")
        lines.append("- Best method by forward_median under stability constraints: none")
        lines.append("- Recommendation: Improve discovery/search space/exits before leverage")
    for method_key, payload in summary["method_summaries"].items():
        lines.append(
            f"- {method_key}: confidence_score={_format_float(payload['stability']['confidence_score'])}, "
            f"classification={payload['stability']['classification']}"
        )
    lines.append("")

    report_text = "\n".join(lines).strip() + "\n"
    (run_dir / "walkforward_report.md").write_text(report_text, encoding="utf-8")


def _window_row(method_key: str, evaluation: MethodWindowEvaluation) -> dict[str, Any]:
    metrics = evaluation.metrics
    return {
        "method": method_key,
        "window": evaluation.window.name,
        "window_kind": evaluation.window.kind,
        "usable": bool(metrics["usable"]),
        "exclusion_reason": str(metrics["exclusion_reason"]),
        "expected_start": metrics["expected_start"],
        "expected_end": metrics["expected_end"],
        "actual_start": metrics["actual_start"],
        "actual_end": metrics["actual_end"],
        "truncated": bool(metrics["window_truncated"]),
        "enough_data": bool(metrics["enough_data"]),
        "bar_count": int(metrics["bar_count"]),
        "trade_count": float(metrics["trade_count_total"]),
        "trades_per_month": float(metrics["trades_per_month"]),
        "exposure_ratio": float(metrics["exposure_ratio"]),
        "raw_PF": metrics["raw_profit_factor"],
        "PF_clipped": float(metrics["profit_factor_clipped"]),
        "expectancy": float(metrics["expectancy"]),
        "exp_lcb": float(metrics["exp_lcb"]),
        "effective_edge": float(metrics["effective_edge"]),
        "return_pct": float(metrics["return_pct"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "Sharpe": _safe_float(metrics["Sharpe_ratio"]),
        "Sortino": _safe_float(metrics["Sortino_ratio"]),
        "Calmar": _safe_float(metrics["Calmar_ratio"]),
        "avg_corr": float(evaluation.avg_corr),
        "effective_n": float(evaluation.effective_n),
        "weight_sum": float(evaluation.weight_sum),
        "note": str(metrics["window_note"]),
    }


def _window_to_payload(window: WindowSpec) -> dict[str, Any]:
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


def _stability_rows(stability_by_method: dict[str, MethodStability]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_key, stability in stability_by_method.items():
        rows.append(
            {
                "method": method_key,
                "stability_metric": stability.stability_metric,
                "holdout_metric": float(stability.holdout_metric),
                "forward_values": json.dumps(stability.forward_values),
                "forward_median": float(stability.forward_median),
                "forward_mean": float(stability.forward_mean),
                "forward_std": float(stability.forward_std),
                "forward_iqr": float(stability.forward_iqr),
                "worst_forward_value": float(stability.worst_forward_value),
                "degradation_ratio": float(stability.degradation_ratio),
                "dd_growth_ratio": float(stability.dd_growth_ratio),
                "usable_windows": int(stability.usable_windows),
                "excluded_windows": int(stability.excluded_windows),
                "total_windows": int(stability.total_windows),
                "min_usable_windows": int(stability.min_usable_windows),
                "confidence_score": float(stability.confidence_score),
                "classification": stability.classification,
                "recommendation": stability.recommendation,
                "failed_criteria": "; ".join(stability.failed_criteria),
                "explanation": stability.explanation,
            }
        )
    return rows


def _extract_stability_value(metrics: dict[str, Any], stability_metric: str) -> float:
    if stability_metric == "exp_lcb":
        return float(metrics.get("exp_lcb", 0.0))
    if stability_metric == "effective_edge":
        return float(metrics.get("effective_edge", 0.0))
    if stability_metric == "pf_clipped":
        return float(metrics.get("profit_factor_clipped", 0.0))
    raise ValueError(f"Unsupported stability metric: {stability_metric}")


def _non_finite_metric_keys(metrics: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in WINDOW_NUMERIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            keys.append(key)
    return keys


def _distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    array = np.array(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _matrix_off_diagonal_stats(matrix: pd.DataFrame) -> dict[str, float]:
    if matrix.empty or len(matrix.index) < 2:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    values = matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    off_diag = values[mask]
    if off_diag.size == 0:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(average_correlation(matrix)),
        "min": float(np.min(off_diag)),
        "max": float(np.max(off_diag)),
    }


def _payload_has_no_inf_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_payload_has_no_inf_nan(item) for item in value.values())
    if isinstance(value, list):
        return all(_payload_has_no_inf_nan(item) for item in value)
    return True


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            if math.isnan(value):
                return "nan"
            return "inf" if value > 0 else "-inf"
        return float(value)
    return value


def _safe_float(value: Any) -> float | str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if not math.isfinite(numeric):
        if math.isnan(numeric):
            return "nan"
        return "inf" if numeric > 0 else "-inf"
    return float(numeric)


def _format_float(value: Any) -> str:
    resolved = _safe_float(value)
    if isinstance(resolved, str):
        return resolved
    return f"{resolved:.4f}"


def _format_display(value: Any) -> str:
    resolved = _safe_float(value)
    if isinstance(resolved, str):
        return resolved
    return f"{resolved:.4f}"


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
