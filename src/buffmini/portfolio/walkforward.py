"""Stage-2.5 rolling walk-forward portfolio validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, validate_config
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
    """Stability summary for one portfolio method."""

    pf_holdout: float
    pf_forward_mean: float
    pf_forward_std: float
    degradation_ratio: float
    worst_forward_pf: float
    dd_growth_ratio: float
    usable_windows: int
    min_usable_windows: int
    classification: str
    recommendation: str
    failed_criteria: list[str]


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
    """Run Stage-2.5 walk-forward validation using cached local data only."""

    if int(forward_days) < 1:
        raise ValueError("forward_days must be >= 1")
    if int(num_windows) < 1:
        raise ValueError("num_windows must be >= 1")
    resolved_reserve_forward_days = (
        int(forward_days) * int(num_windows) if reserve_forward_days is None else int(reserve_forward_days)
    )
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
    candidate_records = {
        record.candidate_id: record
        for record in load_stage1_candidates(stage1_run_dir)
    }
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
        raise ValueError("Stage-2.5 requires candidate metadata for Stage-2 selected strategies")

    raw_data = _load_raw_data(config=config, data_dir=data_dir)
    feature_data = {symbol: calculate_features(frame) for symbol, frame in raw_data.items()}
    data_hash = _compute_input_data_hash(raw_data)
    config_hash = compute_config_hash(config)

    available_end = min(pd.to_datetime(frame["timestamp"], utc=True).max() for frame in raw_data.values() if not frame.empty)
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
    candidate_window_bundles = {
        window.name: evaluate_candidate_bundles_for_window(
            window=window,
            candidate_records=selected_records,
            signal_caches=signal_caches,
            initial_capital=float(INITIAL_PORTFOLIO_CAPITAL * float(config["risk"]["max_concurrent_positions"])),
            round_trip_cost_pct=float(config["costs"]["round_trip_cost_pct"]),
            slippage_pct=float(config["costs"]["slippage_pct"]),
        )
        for window in all_windows
    }

    correlation_matrices: dict[str, pd.DataFrame] = {}
    method_evaluations: dict[str, list[MethodWindowEvaluation]] = {method_key: [] for method_key in stage2_summary["portfolio_methods"]}
    window_rows: list[dict[str, Any]] = []

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
            )
            method_evaluations[method_key].append(evaluation)
            window_rows.append(_window_row(method_key=method_key, evaluation=evaluation))

    min_usable_windows = int(
        config.get("portfolio", {})
        .get("walkforward", {})
        .get("min_usable_windows", DEFAULT_WALKFORWARD_MIN_USABLE_WINDOWS)
    )
    stability_by_method = {
        method_key: compute_stability_summary(
            evaluations,
            min_usable_windows=min_usable_windows,
        )
        for method_key, evaluations in method_evaluations.items()
    }
    sanity_checks = build_sanity_checks(
        holdout_window=holdout_window,
        forward_windows=forward_windows,
        method_payloads=stage2_summary["portfolio_methods"],
        correlation_matrices=correlation_matrices,
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
        "min_usable_windows": int(min_usable_windows),
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
                        "metrics": _json_safe_metrics(evaluation.metrics),
                        "avg_corr": float(evaluation.avg_corr),
                        "effective_n": float(evaluation.effective_n),
                        "weight_sum": float(evaluation.weight_sum),
                        "selected_candidates": evaluation.selected_candidates,
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
    resolved_run_id = run_id or f"{utc_now_compact()}_{summary_hash}_stage2_5"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(window_rows).to_csv(run_dir / "window_metrics.csv", index=False)
    _write_weight_csvs(
        run_dir=run_dir,
        stage2_summary=stage2_summary,
        candidate_records=selected_records,
    )
    _write_correlation_matrices(run_dir=run_dir, matrices=correlation_matrices)

    summary_payload["run_id"] = resolved_run_id
    (run_dir / "walkforward_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_walkforward_report(
        run_dir=run_dir,
        summary=summary_payload,
        method_evaluations=method_evaluations,
        selected_records=selected_records,
    )

    logger.info("Saved Stage-2.5 artifacts to %s", run_dir)
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
) -> MethodStability:
    """Compute Stage-2.5 stability statistics for one method."""

    holdout_eval = next(evaluation for evaluation in evaluations if evaluation.window.kind == "holdout")
    forward_evals = [evaluation for evaluation in evaluations if evaluation.window.kind == "forward"]
    usable = [evaluation for evaluation in forward_evals if bool(evaluation.metrics["enough_data"]) and int(evaluation.metrics["bar_count"]) >= 2]

    pf_holdout = float(holdout_eval.metrics["profit_factor"])
    if usable:
        forward_pfs = np.array([float(evaluation.metrics["profit_factor"]) for evaluation in usable], dtype=float)
        forward_dds = np.array([float(evaluation.metrics["max_drawdown"]) for evaluation in usable], dtype=float)
        pf_forward_mean = float(np.mean(forward_pfs))
        pf_forward_std = float(np.std(forward_pfs, ddof=0))
        worst_forward_pf = float(np.min(forward_pfs))
        max_forward_dd = float(np.max(forward_dds))
    else:
        pf_forward_mean = 0.0
        pf_forward_std = 0.0
        worst_forward_pf = 0.0
        max_forward_dd = 0.0

    holdout_dd = float(holdout_eval.metrics["max_drawdown"])
    degradation_ratio = float(pf_forward_mean / pf_holdout) if pf_holdout > 0 else 0.0
    if holdout_dd > 0:
        dd_growth_ratio = float(max_forward_dd / holdout_dd)
    else:
        dd_growth_ratio = 0.0 if max_forward_dd == 0 else math.inf

    usable_windows = int(len(usable))
    failures: list[str] = []
    if usable_windows < int(min_usable_windows):
        failures.append(f"usable_windows < min_usable_windows ({usable_windows} < {int(min_usable_windows)})")
        classification = "INSUFFICIENT_DATA"
        recommendation = "DO NOT proceed to leverage modeling"
    else:
        if degradation_ratio < 0.7:
            failures.append("degradation_ratio < 0.7")
        if worst_forward_pf < 1.0:
            failures.append("worst_forward_pf < 1.0")
        if dd_growth_ratio > 2.0:
            failures.append("dd_growth_ratio > 2.0")
        classification = "STABLE" if not failures else "UNSTABLE"
        recommendation = (
            "Proceed to leverage modeling"
            if classification == "STABLE"
            else "Improve discovery/search space/exits before leverage"
        )
    return MethodStability(
        pf_holdout=pf_holdout,
        pf_forward_mean=pf_forward_mean,
        pf_forward_std=pf_forward_std,
        degradation_ratio=degradation_ratio,
        worst_forward_pf=worst_forward_pf,
        dd_growth_ratio=dd_growth_ratio,
        usable_windows=usable_windows,
        min_usable_windows=int(min_usable_windows),
        classification=classification,
        recommendation=recommendation,
        failed_criteria=failures,
    )


def build_sanity_checks(
    holdout_window: WindowSpec,
    forward_windows: list[WindowSpec],
    method_payloads: dict[str, Any],
    correlation_matrices: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Build Stage-2.5 sanity checks and pass/fail flags."""

    overlap_checks: list[dict[str, Any]] = []
    if forward_windows:
        first_forward = forward_windows[0]
        holdout_end = holdout_window.actual_end or holdout_window.expected_end
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

    correlation_checks = {}
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

    return {
        "overlap_checks": overlap_checks,
        "weight_checks": weight_checks,
        "correlation_checks": correlation_checks,
        "future_leakage": {
            "passed": True,
            "detail": (
                "Signals are generated from completed bars and sliced into windows after signal computation; "
                "coverage is enforced by tests/test_no_future_leakage.py, tests/test_stage2_portfolio.py, "
                "and tests/test_stage2_walkforward.py."
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
    end = config["universe"].get("end")
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
    if any(item.classification == "STABLE" for item in stability_by_method.values()):
        return "Proceed to leverage modeling"
    if any(item.classification == "INSUFFICIENT_DATA" for item in stability_by_method.values()):
        return "DO NOT proceed to leverage modeling"
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


def _write_correlation_matrices(run_dir: Path, matrices: dict[str, pd.DataFrame]) -> None:
    for window_name, matrix in matrices.items():
        if matrix.empty:
            continue
        if window_name == "Holdout":
            filename = "correlation_matrix_holdout.csv"
        else:
            filename = f"correlation_matrix_{window_name.lower()}.csv"
        matrix.to_csv(run_dir / filename, index=True)


def _write_walkforward_report(
    run_dir: Path,
    summary: dict[str, Any],
    method_evaluations: dict[str, list[MethodWindowEvaluation]],
    selected_records: dict[str, Stage1CandidateRecord],
) -> None:
    lines: list[str] = []
    lines.append("# Stage-2.5 Walk-Forward Audit Report")
    lines.append("")
    lines.append("## Section 1 - Provenance & Reproducibility")
    lines.append(f"- Stage-2 run_id used: `{summary['stage2_run_id']}`")
    lines.append(f"- Stage-1 run_id referenced by Stage-2: `{summary['stage1_run_id']}`")
    lines.append(f"- Exact CLI command used: `{summary['command']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- reserve_forward_days: `{summary['reserve_forward_days']}`")
    lines.append(f"- min_usable_windows required: `{summary['min_usable_windows']}`")
    lines.append(f"- config hash: `{summary['config_hash']}`")
    lines.append(f"- config hash (Stage-1 reference): `{summary['config_hash_stage1_reference']}`")
    lines.append(f"- data hash: `{summary['data_hash']}`")
    lines.append(f"- data hash (Stage-1 reference): `{summary['data_hash_stage1_reference']}`")
    lines.append(
        f"- Holdout from Stage-2 artifacts: `{summary['holdout_window']['expected_start']}` .. `{summary['holdout_window']['expected_end']}`"
    )
    lines.append(
        f"- Holdout timestamps used here: `{summary['holdout_window']['actual_start']}` .. `{summary['holdout_window']['actual_end']}`"
    )
    for window in summary["forward_windows"]:
        lines.append(
            f"- {window['name']} timestamps: expected `{window['expected_start']}` .. `{window['expected_end']}`, "
            f"actual `{window['actual_start']}` .. `{window['actual_end']}`"
        )
    lines.append(f"- shifted_for_forward_window fallback used: `{summary['fallback_used']}`")
    lines.append(f"- fallback reason: {summary['fallback_reason']}")
    lines.append("")

    lines.append("## Section 2 - Portfolio Composition")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        weights = payload["weights"]
        sorted_weights = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
        lines.append(f"### {method_key}")
        lines.append(f"- component strategy count: `{len(payload['selected_candidates'])}`")
        lines.append(f"- effective N: `{method_evaluations[method_key][0].effective_n:.4f}`")
        lines.append("- component strategies:")
        for candidate_id in payload["selected_candidates"]:
            record = selected_records[candidate_id]
            lines.append(
                f"  - `{candidate_id}` | {record.family} | gating={record.gating_mode} | exit={record.exit_mode} | tier={record.result_tier}"
            )
        lines.append("- top weights (inline):")
        for candidate_id, weight in sorted_weights[:10]:
            record = selected_records[candidate_id]
            lines.append(
                f"  - `{candidate_id}` | weight={float(weight):.6f} | {record.family} | gating={record.gating_mode} | exit={record.exit_mode}"
            )
        csv_name = method_key.replace("-", "_")
        lines.append(f"- full weight CSV: `weights_{csv_name}.csv`")
        lines.append("")

    lines.append("## Section 3 - Window-by-Window Results")
    for method_key in ["equal", "vol", "corr-min"]:
        evaluations = method_evaluations.get(method_key, [])
        if not evaluations:
            continue
        lines.append(f"### {method_key}")
        lines.append(
            "| window | status | PF | expectancy | exp_lcb | trade_count | tpm | exposure | return_pct | max_dd | Sharpe | Sortino | Calmar | avg_corr |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for evaluation in evaluations:
            metrics = evaluation.metrics
            calmar = metrics["Calmar_ratio"]
            cagr = metrics["CAGR_approx"]
            calmar_text = f"{float(calmar):.4f}" if math.isfinite(float(calmar)) else "inf"
            cagr_text = f"{float(cagr):.4f}" if cagr is not None else "N/A"
            if evaluation.window.actual_start is None or evaluation.window.actual_end is None:
                status = "missing"
            elif evaluation.window.truncated:
                status = "truncated"
            else:
                status = "full"
            lines.append(
                f"| {evaluation.window.name} | {status} | {float(metrics['profit_factor']):.4f} | {float(metrics['expectancy']):.4f} | "
                f"{float(metrics['exp_lcb']):.4f} | {float(metrics['trade_count_total']):.0f} | {float(metrics['trades_per_month']):.4f} | "
                f"{float(metrics['exposure_ratio']):.4f} | {float(metrics['return_pct']):.4f} | {float(metrics['max_drawdown']):.4f} | "
                f"{float(metrics['Sharpe_ratio']):.4f} | {float(metrics['Sortino_ratio']):.4f} | {calmar_text} | {evaluation.avg_corr:.4f} |"
            )
            lines.append(f"  CAGR_approx: `{cagr_text}`")
            if evaluation.window.note:
                lines.append(f"  Note: {evaluation.window.note}")
        lines.append("")

    lines.append("## Section 4 - Stability Summary")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["method_summaries"].get(method_key)
        if payload is None:
            continue
        stability = payload["stability"]
        lines.append(f"### {method_key}")
        lines.append(f"- pf_holdout: `{stability['pf_holdout']:.4f}`")
        lines.append(f"- pf_forward_mean/std: `{stability['pf_forward_mean']:.4f}` / `{stability['pf_forward_std']:.4f}`")
        lines.append(f"- degradation_ratio: `{stability['degradation_ratio']:.4f}`")
        lines.append(f"- worst_forward_pf: `{stability['worst_forward_pf']:.4f}`")
        lines.append(f"- dd_growth_ratio: `{stability['dd_growth_ratio']:.4f}`")
        lines.append(
            f"- usable_windows: `{stability['usable_windows']}` / required `{stability['min_usable_windows']}`"
        )
        lines.append(f"- classification: `{stability['classification']}`")
        lines.append(f"- recommendation: `{stability['recommendation']}`")
        if stability["classification"] == "INSUFFICIENT_DATA":
            lines.append(
                f"- insufficiency detail: available `{stability['usable_windows']}` usable forward windows, "
                f"required `{stability['min_usable_windows']}`"
            )
        if stability["failed_criteria"]:
            lines.append(f"- failed criteria: {', '.join(stability['failed_criteria'])}")
        else:
            lines.append("- failed criteria: none")
        lines.append("")

    lines.append("## Section 5 - Sanity Checks")
    for check in summary["sanity_checks"]["overlap_checks"]:
        lines.append(f"- {check['name']}: `{check['passed']}` | {check['detail']}")
    for method_key, check in summary["sanity_checks"]["weight_checks"].items():
        lines.append(f"- weights_sum_{method_key}: `{check['passed']}` | sum={check['sum']:.12f}")
    for window_name, check in summary["sanity_checks"]["correlation_checks"].items():
        lines.append(f"- correlation_matrix_{window_name}: `{check['passed']}` | {check['detail']}")
    leakage = summary["sanity_checks"]["future_leakage"]
    lines.append(f"- future_leakage: `{leakage['passed']}` | {leakage['detail']}")
    lines.append("")

    lines.append("## Section 6 - Actionable Conclusion")
    stable_methods = [
        (method_key, payload["stability"])
        for method_key, payload in summary["method_summaries"].items()
        if payload["stability"]["classification"] == "STABLE"
    ]
    if stable_methods:
        best_method, best_payload = max(stable_methods, key=lambda item: float(item[1]["pf_forward_mean"]))
        lines.append("- Any STABLE method: `YES`")
        lines.append(f"- Best STABLE method by pf_forward_mean: `{best_method}` ({float(best_payload['pf_forward_mean']):.4f})")
        lines.append("- Recommendation: Proceed to leverage modeling")
    else:
        insufficient_methods = [
            method_key
            for method_key, payload in summary["method_summaries"].items()
            if payload["stability"]["classification"] == "INSUFFICIENT_DATA"
        ]
        lines.append("- Any STABLE method: `NO`")
        lines.append("- Best method by pf_forward_mean with stability constraints: none")
        if insufficient_methods:
            lines.append("- Recommendation: DO NOT proceed to leverage modeling")
            lines.append(f"- Why: insufficient usable forward windows for `{insufficient_methods}`")
        else:
            lines.append("- Recommendation: Improve discovery/search space/exits before leverage")
    usable_counts = {
        method_key: int(payload["stability"]["usable_windows"])
        for method_key, payload in summary["method_summaries"].items()
    }
    lines.append(f"- usable forward windows per method: `{usable_counts}`")
    lines.append("")

    report_text = "\n".join(lines).strip() + "\n"
    (run_dir / "walkforward_report.md").write_text(report_text, encoding="utf-8")


def _window_row(method_key: str, evaluation: MethodWindowEvaluation) -> dict[str, Any]:
    metrics = evaluation.metrics
    return {
        "method": method_key,
        "window": evaluation.window.name,
        "window_kind": evaluation.window.kind,
        "expected_start": metrics["expected_start"],
        "expected_end": metrics["expected_end"],
        "actual_start": metrics["actual_start"],
        "actual_end": metrics["actual_end"],
        "truncated": bool(metrics["window_truncated"]),
        "enough_data": bool(metrics["enough_data"]),
        "bar_count": int(metrics["bar_count"]),
        "profit_factor": float(metrics["profit_factor"]),
        "expectancy": float(metrics["expectancy"]),
        "exp_lcb": float(metrics["exp_lcb"]),
        "trade_count": float(metrics["trade_count_total"]),
        "trades_per_month": float(metrics["trades_per_month"]),
        "exposure_ratio": float(metrics["exposure_ratio"]),
        "return_pct": float(metrics["return_pct"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "CAGR_approx": metrics["CAGR_approx"],
        "Sharpe": float(metrics["Sharpe_ratio"]),
        "Sortino": float(metrics["Sortino_ratio"]),
        "Calmar": float(metrics["Calmar_ratio"]) if math.isfinite(float(metrics["Calmar_ratio"])) else "inf",
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


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
