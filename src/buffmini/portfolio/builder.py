"""Stage-2 portfolio construction and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals
from buffmini.config import get_universe_end
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.discovery.generator import Candidate, candidate_to_strategy_spec
from buffmini.portfolio.correlation import (
    DEFAULT_CORR_MIN_SUBSET_SIZE,
    average_correlation,
    build_correlation_matrix,
    effective_number_of_strategies,
    normalize_weights,
    select_correlation_minimized_subset,
)
from buffmini.portfolio.metrics import INITIAL_PORTFOLIO_CAPITAL, build_portfolio_equity, compute_portfolio_metrics
from buffmini.types import ConfigDict
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)

_ALLOWED_METHODS = {"equal", "vol", "corr-min", "corr", "min", "all"}


@dataclass(frozen=True)
class Stage1CandidateRecord:
    """Stage-1 candidate metadata required for Stage-2 reconstruction."""

    candidate_id: str
    result_tier: str
    strategy_name: str
    family: str
    gating_mode: str
    exit_mode: str
    params: dict[str, Any]
    holdout_months_used: int
    effective_edge: float
    exact_holdout_range: str | None = None

    def to_candidate(self) -> Candidate:
        return Candidate(
            candidate_id=self.candidate_id,
            family=self.family,
            gating_mode=self.gating_mode,
            exit_mode=self.exit_mode,
            params=dict(self.params),
        )


@dataclass(frozen=True)
class EvaluationWindow:
    """Resolved candidate evaluation windows."""

    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    forward_end: pd.Timestamp
    window_mode: str


@dataclass
class CandidateWindowResult:
    """Candidate-level holdout and forward artifacts for Stage-2."""

    record: Stage1CandidateRecord
    window: EvaluationWindow
    holdout_returns: pd.Series
    holdout_equity: pd.Series
    holdout_exposure: pd.Series
    holdout_trades: pd.DataFrame
    forward_returns: pd.Series
    forward_equity: pd.Series
    forward_exposure: pd.Series
    forward_trades: pd.DataFrame


def run_stage2_portfolio(
    stage1_run_id: str,
    method: str = "all",
    forward_days: int = 30,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Run Stage-2 portfolio construction from an existing Stage-1 run."""

    resolved_method = _normalize_method(method)
    if int(forward_days) < 1:
        raise ValueError("forward_days must be >= 1")

    stage1_run_dir = _resolve_stage1_run_dir(stage1_run_id=stage1_run_id, runs_dir=runs_dir)
    config = _load_stage1_config(stage1_run_dir=stage1_run_dir)
    summary = _load_json(stage1_run_dir / "summary.json")
    candidate_records = load_stage1_candidates(stage1_run_dir=stage1_run_dir)
    if not candidate_records:
        raise ValueError("Stage-2 requires at least one Tier A or Tier B candidate")

    feature_data = _load_feature_data(config=config, data_dir=data_dir)
    initial_capital = float(INITIAL_PORTFOLIO_CAPITAL * float(config["risk"]["max_concurrent_positions"]))
    round_trip_cost_pct = float(config["costs"]["round_trip_cost_pct"])
    slippage_pct = float(config["costs"]["slippage_pct"])

    candidate_results = {
        record.candidate_id: evaluate_candidate_windows(
            record=record,
            feature_data=feature_data,
            initial_capital=initial_capital,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            forward_days=int(forward_days),
        )
        for record in candidate_records
    }

    holdout_return_map = {
        candidate_id: result.holdout_returns
        for candidate_id, result in candidate_results.items()
        if not result.holdout_returns.empty
    }
    correlation_matrix = build_correlation_matrix(holdout_return_map)
    overall_average_correlation = average_correlation(correlation_matrix)

    methods = _select_methods(resolved_method)
    method_summaries: dict[str, dict[str, Any]] = {}
    for method_key in methods:
        weights = _build_method_weights(
            method_key=method_key,
            candidate_results=candidate_results,
            correlation_matrix=correlation_matrix,
        )
        portfolio_result = _build_portfolio_result(
            method_key=method_key,
            candidate_results=candidate_results,
            weights=weights,
            correlation_matrix=correlation_matrix,
        )
        method_summaries[method_key] = portfolio_result

    stage2_payload = {
        "stage1_run_id": stage1_run_id,
        "forward_days": int(forward_days),
        "method": resolved_method,
        "candidate_count_tier_A": int(sum(1 for record in candidate_records if record.result_tier == "Tier A")),
        "candidate_count_tier_B": int(sum(1 for record in candidate_records if record.result_tier == "Tier B")),
        "candidate_count_total": int(len(candidate_records)),
        "average_correlation": float(overall_average_correlation),
        "correlation_subset_size": int(min(DEFAULT_CORR_MIN_SUBSET_SIZE, len(candidate_records))),
        "portfolio_methods": method_summaries,
        "window_modes": sorted({result.window.window_mode for result in candidate_results.values()}),
        "window_mode_note": (
            "Exact Stage-1 holdout ended at the latest available bar; "
            "Stage-2 shifted holdout back by forward_days to preserve a non-overlapping forward window."
            if "shifted_for_forward_window" in {result.window.window_mode for result in candidate_results.values()}
            else "Forward window starts immediately after the exact Stage-1 holdout."
        ),
        "stage1_result_thresholds": summary.get("result_thresholds", {}),
        "round_trip_cost_pct": round_trip_cost_pct,
        "slippage_pct": slippage_pct,
    }

    run_id = f"{utc_now_compact()}_{stable_hash(stage2_payload, length=12)}_stage2"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    correlation_matrix.to_csv(run_dir / "correlation_matrix.csv", index=True)
    for method_key, payload in method_summaries.items():
        _write_portfolio_csv(run_dir=run_dir, method_key=method_key, payload=payload)

    summary_payload = {
        "run_id": run_id,
        **_json_safe_stage2_payload(stage2_payload),
    }
    (run_dir / "portfolio_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_portfolio_report(run_dir=run_dir, summary=summary_payload)

    logger.info("Saved Stage-2 artifacts to %s", run_dir)
    return run_dir


def load_stage1_candidates(stage1_run_dir: Path) -> list[Stage1CandidateRecord]:
    """Load Tier A and Tier B candidates from a Stage-1 run."""

    strategies = _load_json(stage1_run_dir / "strategies.json")
    strategy_ranges = {
        str(item["candidate_id"]): str((item.get("metrics_holdout") or {}).get("date_range", "n/a"))
        for item in strategies
    }

    candidate_dir = stage1_run_dir / "candidates"
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(candidate_dir.glob("strategy_*.json")):
        payload = _load_json(path)
        candidate_payloads[str(payload["candidate_id"])] = payload

    records: list[Stage1CandidateRecord] = []
    for csv_name, tier_name in [("tier_A_candidates.csv", "Tier A"), ("tier_B_candidates.csv", "Tier B")]:
        path = stage1_run_dir / csv_name
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            continue
        if "candidate_id" not in frame.columns:
            continue
        for row in frame.to_dict(orient="records"):
            candidate_id = str(row["candidate_id"])
            payload = candidate_payloads.get(candidate_id)
            if payload is None:
                raise ValueError(f"Missing candidate artifact for {candidate_id}")
            records.append(
                Stage1CandidateRecord(
                    candidate_id=candidate_id,
                    result_tier=tier_name,
                    strategy_name=str(payload["strategy_name"]),
                    family=str(payload["strategy_family"]),
                    gating_mode=str(payload["gating"]),
                    exit_mode=str(payload["exit_mode"]),
                    params=dict(payload["parameters"]),
                    holdout_months_used=int(payload["holdout_months_used"]),
                    effective_edge=float(payload["effective_edge"]),
                    exact_holdout_range=_clean_range(strategy_ranges.get(candidate_id)),
                )
            )
    return records


def evaluate_candidate_windows(
    record: Stage1CandidateRecord,
    feature_data: dict[str, pd.DataFrame],
    initial_capital: float,
    round_trip_cost_pct: float,
    slippage_pct: float,
    forward_days: int,
) -> CandidateWindowResult:
    """Reconstruct holdout and forward windows for one Stage-1 candidate."""

    candidate = record.to_candidate()
    signal_cache = build_candidate_signal_cache(candidate=candidate, feature_data=feature_data)
    window = resolve_evaluation_window(
        record=record,
        feature_data=feature_data,
        forward_days=int(forward_days),
    )

    holdout_frames = {
        symbol: slice_time_window(
            frame=frame,
            start=window.holdout_start,
            end=window.holdout_end,
            start_inclusive=True,
            end_inclusive=True,
        )
        for symbol, frame in signal_cache.items()
    }
    forward_frames = {
        symbol: slice_time_window(
            frame=frame,
            start=window.holdout_end,
            end=window.forward_end,
            start_inclusive=False,
            end_inclusive=True,
        )
        for symbol, frame in signal_cache.items()
    }

    holdout_bundle = _run_candidate_bundle(
        candidate=candidate,
        frames=holdout_frames,
        initial_capital=initial_capital,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )
    forward_bundle = _run_candidate_bundle(
        candidate=candidate,
        frames=forward_frames,
        initial_capital=initial_capital,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )

    return CandidateWindowResult(
        record=record,
        window=window,
        holdout_returns=holdout_bundle["returns"],
        holdout_equity=holdout_bundle["equity"],
        holdout_exposure=holdout_bundle["exposure"],
        holdout_trades=holdout_bundle["trades"],
        forward_returns=forward_bundle["returns"],
        forward_equity=forward_bundle["equity"],
        forward_exposure=forward_bundle["exposure"],
        forward_trades=forward_bundle["trades"],
    )


def build_candidate_signal_cache(
    candidate: Candidate,
    feature_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compute signals once on the full feature set for a candidate."""

    strategy = candidate_to_strategy_spec(candidate)
    cached: dict[str, pd.DataFrame] = {}
    for symbol, frame in feature_data.items():
        prepared = frame.copy().sort_values("timestamp").reset_index(drop=True)
        prepared["signal"] = generate_signals(prepared, strategy, gating_mode=candidate.gating_mode)
        cached[symbol] = prepared
    return cached


def resolve_evaluation_window(
    record: Stage1CandidateRecord,
    feature_data: dict[str, pd.DataFrame],
    forward_days: int,
) -> EvaluationWindow:
    """Resolve a non-overlapping holdout/forward evaluation window."""

    if int(forward_days) < 1:
        raise ValueError("forward_days must be >= 1")

    common_end = min(pd.to_datetime(frame["timestamp"], utc=True).max() for frame in feature_data.values() if not frame.empty)
    exact_range = _parse_date_range(record.exact_holdout_range)
    forward_delta = pd.Timedelta(days=int(forward_days))

    if exact_range is not None:
        exact_start, exact_end = exact_range
        if exact_end + forward_delta <= common_end:
            return EvaluationWindow(
                holdout_start=exact_start,
                holdout_end=exact_end,
                forward_end=exact_end + forward_delta,
                window_mode="exact_stage1_holdout",
            )

    shifted_holdout_end = common_end - forward_delta
    if pd.isna(shifted_holdout_end):
        raise ValueError("Unable to derive shifted holdout end for Stage-2")
    shifted_holdout_start = shifted_holdout_end - pd.DateOffset(months=int(record.holdout_months_used))
    return EvaluationWindow(
        holdout_start=_ensure_utc(shifted_holdout_start),
        holdout_end=_ensure_utc(shifted_holdout_end),
        forward_end=_ensure_utc(common_end),
        window_mode="shifted_for_forward_window",
    )


def slice_time_window(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    start_inclusive: bool,
    end_inclusive: bool,
) -> pd.DataFrame:
    """Slice a timestamped frame without overlap."""

    if frame.empty:
        return frame.copy()

    data = frame.copy().sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    mask = pd.Series(True, index=data.index)
    start_ts = _ensure_utc(start)
    end_ts = _ensure_utc(end)
    mask &= (timestamps >= start_ts) if start_inclusive else (timestamps > start_ts)
    mask &= (timestamps <= end_ts) if end_inclusive else (timestamps < end_ts)
    return data.loc[mask].reset_index(drop=True)


def build_equal_weights(candidate_ids: list[str]) -> dict[str, float]:
    """Build equal portfolio weights."""

    if not candidate_ids:
        return {}
    return normalize_weights({candidate_id: 1.0 for candidate_id in candidate_ids})


def build_volatility_weights(candidate_results: dict[str, CandidateWindowResult]) -> dict[str, float]:
    """Build inverse-volatility weights from holdout returns."""

    raw: dict[str, float] = {}
    for candidate_id, result in candidate_results.items():
        volatility = float(result.holdout_returns.std(ddof=0)) if not result.holdout_returns.empty else 0.0
        raw[candidate_id] = 0.0 if volatility <= 0 else 1.0 / volatility
    return normalize_weights(raw)


def build_portfolio_return_series(
    series_by_candidate: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Combine aligned candidate return series into a portfolio return series."""

    if not weights:
        return pd.Series(dtype=float)
    active_series = {
        candidate_id: series
        for candidate_id, series in series_by_candidate.items()
        if candidate_id in weights
    }
    if not active_series:
        return pd.Series(dtype=float)
    aligned = pd.concat(active_series, axis=1).sort_index().fillna(0.0)
    return aligned.mul(pd.Series(weights), axis=1).sum(axis=1).astype(float)


def _build_method_weights(
    method_key: str,
    candidate_results: dict[str, CandidateWindowResult],
    correlation_matrix: pd.DataFrame,
) -> dict[str, float]:
    candidate_ids = list(candidate_results.keys())
    if method_key == "equal":
        return build_equal_weights(candidate_ids)
    if method_key == "vol":
        return build_volatility_weights(candidate_results)
    if method_key == "corr-min":
        selected = select_correlation_minimized_subset(
            candidate_ids=candidate_ids,
            correlation_matrix=correlation_matrix,
            effective_edge={candidate_id: result.record.effective_edge for candidate_id, result in candidate_results.items()},
            subset_size=min(DEFAULT_CORR_MIN_SUBSET_SIZE, len(candidate_ids)),
        )
        return normalize_weights({candidate_id: 1.0 for candidate_id in selected})
    raise ValueError(f"Unsupported Stage-2 method: {method_key}")


def _build_portfolio_result(
    method_key: str,
    candidate_results: dict[str, CandidateWindowResult],
    weights: dict[str, float],
    correlation_matrix: pd.DataFrame,
) -> dict[str, Any]:
    selected_ids = [candidate_id for candidate_id, weight in weights.items() if float(weight) > 0]
    holdout_returns = build_portfolio_return_series(
        {candidate_id: result.holdout_returns for candidate_id, result in candidate_results.items()},
        weights,
    )
    forward_returns = build_portfolio_return_series(
        {candidate_id: result.forward_returns for candidate_id, result in candidate_results.items()},
        weights,
    )
    holdout_exposure = build_portfolio_return_series(
        {candidate_id: result.holdout_exposure for candidate_id, result in candidate_results.items()},
        weights,
    )
    forward_exposure = build_portfolio_return_series(
        {candidate_id: result.forward_exposure for candidate_id, result in candidate_results.items()},
        weights,
    )
    holdout_equity = build_portfolio_equity(holdout_returns)
    forward_equity = build_portfolio_equity(forward_returns)

    holdout_trade_pnls = _combine_weighted_trade_pnls(
        {candidate_id: result.holdout_trades for candidate_id, result in candidate_results.items()},
        weights,
    )
    forward_trade_pnls = _combine_weighted_trade_pnls(
        {candidate_id: result.forward_trades for candidate_id, result in candidate_results.items()},
        weights,
    )

    holdout_metrics = compute_portfolio_metrics(
        returns=holdout_returns,
        equity=holdout_equity,
        trade_pnls=holdout_trade_pnls,
        exposure=holdout_exposure,
    )
    forward_metrics = compute_portfolio_metrics(
        returns=forward_returns,
        equity=forward_equity,
        trade_pnls=forward_trade_pnls,
        exposure=forward_exposure,
    )

    selected_corr = (
        correlation_matrix.reindex(index=selected_ids, columns=selected_ids)
        if selected_ids and not correlation_matrix.empty
        else pd.DataFrame()
    )
    avg_selected_corr = average_correlation(selected_corr)
    degradation = {
        "profit_factor_delta": float(forward_metrics["profit_factor"] - holdout_metrics["profit_factor"]),
        "return_pct_delta": float(forward_metrics["return_pct"] - holdout_metrics["return_pct"]),
        "max_drawdown_ratio": float(
            forward_metrics["max_drawdown"] / holdout_metrics["max_drawdown"]
        )
        if float(holdout_metrics["max_drawdown"]) > 0
        else 0.0,
    }

    return {
        "method": method_key,
        "weights": {candidate_id: float(weight) for candidate_id, weight in weights.items() if float(weight) > 0},
        "selected_candidates": selected_ids,
        "effective_number_of_strategies": float(effective_number_of_strategies(weights)),
        "average_correlation": float(avg_selected_corr),
        "holdout": holdout_metrics,
        "forward": forward_metrics,
        "degradation": degradation,
        "holdout_series": _series_to_frame(holdout_returns, holdout_equity, holdout_exposure, window="holdout"),
        "forward_series": _series_to_frame(forward_returns, forward_equity, forward_exposure, window="forward"),
    }


def _run_candidate_bundle(
    candidate: Candidate,
    frames: dict[str, pd.DataFrame],
    initial_capital: float,
    round_trip_cost_pct: float,
    slippage_pct: float,
) -> dict[str, Any]:
    strategy = candidate_to_strategy_spec(candidate)
    symbol_equities: dict[str, pd.Series] = {}
    symbol_exposures: dict[str, pd.Series] = {}
    trade_frames: list[pd.DataFrame] = []

    active_symbols = [symbol for symbol, frame in frames.items() if not frame.empty]
    symbol_count = max(1, len(active_symbols))

    for symbol, frame in frames.items():
        if frame.empty:
            continue
        result = run_backtest(
            frame=frame,
            strategy_name=strategy.name,
            symbol=symbol,
            stop_atr_multiple=float(candidate.params["atr_sl_multiplier"]),
            take_profit_atr_multiple=float(candidate.params["atr_tp_multiplier"]),
            max_hold_bars=int(candidate.params["max_holding_bars"]),
            round_trip_cost_pct=float(round_trip_cost_pct),
            slippage_pct=float(slippage_pct),
            initial_capital=float(initial_capital),
            exit_mode=candidate.exit_mode,
            trailing_atr_k=float(candidate.params["trailing_atr_k"]),
        )
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        equity_series = pd.Series(float(initial_capital), index=pd.DatetimeIndex(timestamps), dtype=float)
        if not result.equity_curve.empty:
            curve = result.equity_curve.copy()
            curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
            curve_series = curve.set_index("timestamp")["equity"].astype(float)
            equity_series = curve_series.reindex(equity_series.index).ffill().fillna(float(initial_capital))
        symbol_equities[symbol] = equity_series / float(initial_capital)
        symbol_exposures[symbol] = build_active_exposure_series(timestamps=timestamps, trades=result.trades)
        if not result.trades.empty:
            trades = result.trades.copy()
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
            trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
            trades["candidate_id"] = candidate.candidate_id
            trades["strategy_name"] = strategy.name
            trades["symbol"] = symbol
            trades["candidate_pnl"] = trades["pnl"].astype(float) / float(symbol_count)
            trades["candidate_return_pct"] = trades["candidate_pnl"] / float(initial_capital)
            trade_frames.append(trades)

    if not symbol_equities:
        return {
            "returns": pd.Series(dtype=float),
            "equity": pd.Series(dtype=float),
            "exposure": pd.Series(dtype=float),
            "trades": pd.DataFrame(columns=["candidate_id", "candidate_pnl"]),
        }

    equity_df = pd.concat(symbol_equities, axis=1).sort_index().ffill().fillna(1.0)
    candidate_equity = equity_df.mean(axis=1) * float(initial_capital)
    candidate_returns = candidate_equity.pct_change().fillna(0.0).astype(float)

    exposure_df = pd.concat(symbol_exposures, axis=1).sort_index().fillna(0.0)
    candidate_exposure = exposure_df.mean(axis=1).astype(float)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(
        columns=["candidate_id", "candidate_pnl", "entry_time", "exit_time"]
    )
    if not trades.empty:
        trades = trades.sort_values(["exit_time", "entry_time", "symbol"]).reset_index(drop=True)

    return {
        "returns": candidate_returns,
        "equity": candidate_equity,
        "exposure": candidate_exposure,
        "trades": trades,
    }


def build_active_exposure_series(timestamps: pd.Series | pd.DatetimeIndex, trades: pd.DataFrame) -> pd.Series:
    """Map trades to bar-level active exposure flags."""

    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    exposure = pd.Series(0.0, index=index, dtype=float)
    if trades.empty:
        return exposure

    entry_times = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    exit_times = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    for entry_time, exit_time in zip(entry_times, exit_times, strict=False):
        if pd.isna(entry_time) or pd.isna(exit_time):
            continue
        exposure.loc[(index > entry_time) & (index <= exit_time)] = 1.0
    return exposure


def _combine_weighted_trade_pnls(
    trades_by_candidate: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.Series:
    frames: list[pd.DataFrame] = []
    for candidate_id, trades in trades_by_candidate.items():
        weight = float(weights.get(candidate_id, 0.0))
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


def _series_to_frame(returns: pd.Series, equity: pd.Series, exposure: pd.Series, window: str) -> pd.DataFrame:
    if returns.empty and equity.empty:
        return pd.DataFrame(columns=["timestamp", "window", "portfolio_return", "equity", "exposure"])
    index = returns.index if not returns.empty else equity.index
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(index, utc=True),
            "window": window,
            "portfolio_return": returns.reindex(index).fillna(0.0).astype(float).values,
            "equity": equity.reindex(index).ffill().fillna(INITIAL_PORTFOLIO_CAPITAL).astype(float).values,
            "exposure": exposure.reindex(index).fillna(0.0).astype(float).values,
        }
    )
    return frame


def _write_portfolio_csv(run_dir: Path, method_key: str, payload: dict[str, Any]) -> None:
    filename_map = {
        "equal": "portfolio_equal_weight.csv",
        "vol": "portfolio_vol_weight.csv",
        "corr-min": "portfolio_corr_min.csv",
    }
    frame = pd.concat([payload["holdout_series"], payload["forward_series"]], ignore_index=True)
    frame.to_csv(run_dir / filename_map[method_key], index=False)


def _write_portfolio_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Stage-2 Portfolio Report")
    lines.append("")
    lines.append(f"- stage1_run_id: `{summary['stage1_run_id']}`")
    lines.append(f"- stage2_run_id: `{summary['run_id']}`")
    lines.append(f"- forward_days: `{summary['forward_days']}`")
    lines.append(f"- candidate_count_tier_A: `{summary['candidate_count_tier_A']}`")
    lines.append(f"- candidate_count_tier_B: `{summary['candidate_count_tier_B']}`")
    lines.append(f"- candidate_count_total: `{summary['candidate_count_total']}`")
    lines.append(f"- window_modes: `{summary['window_modes']}`")
    lines.append(f"- window_mode_note: {summary['window_mode_note']}")
    lines.append("")
    lines.append("## Correlation Stats")
    lines.append(f"- average_correlation: `{summary['average_correlation']:.4f}`")
    lines.append(f"- correlation_subset_size: `{summary['correlation_subset_size']}`")
    lines.append("")
    lines.append("## Portfolio Metrics")
    lines.append("| method | selected | eff_n | avg_corr | hold_PF | hold_exp_lcb | hold_max_dd | hold_CAGR | hold_return | hold_exposure | fwd_PF | fwd_exp_lcb | fwd_max_dd | fwd_CAGR | fwd_return | fwd_exposure |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["portfolio_methods"].get(method_key)
        if payload is None:
            continue
        hold = payload["holdout"]
        forward = payload["forward"]
        lines.append(
            f"| {method_key} | {len(payload['selected_candidates'])} | {payload['effective_number_of_strategies']:.4f} | "
            f"{payload['average_correlation']:.4f} | {float(hold['profit_factor']):.4f} | {float(hold['exp_lcb']):.4f} | "
            f"{float(hold['max_drawdown']):.4f} | {float(hold['CAGR']):.4f} | {float(hold['return_pct']):.4f} | "
            f"{float(hold['exposure_ratio']):.4f} | {float(forward['profit_factor']):.4f} | {float(forward['exp_lcb']):.4f} | "
            f"{float(forward['max_drawdown']):.4f} | {float(forward['CAGR']):.4f} | {float(forward['return_pct']):.4f} | "
            f"{float(forward['exposure_ratio']):.4f} |"
        )
    lines.append("")
    lines.append("## Forward vs Holdout")
    for method_key in ["equal", "vol", "corr-min"]:
        payload = summary["portfolio_methods"].get(method_key)
        if payload is None:
            continue
        degradation = payload["degradation"]
        hold = payload["holdout"]
        forward = payload["forward"]
        lines.append(
            f"- {method_key}: PF_delta={float(degradation['profit_factor_delta']):.4f}, "
            f"return_pct_delta={float(degradation['return_pct_delta']):.4f}, "
            f"max_drawdown_ratio={float(degradation['max_drawdown_ratio']):.4f}"
        )
        if float(forward["profit_factor"]) < 1.0:
            lines.append(f"  - WARNING: forward PF < 1 for {method_key}")
        if float(hold["max_drawdown"]) > 0 and float(forward["max_drawdown"]) > float(hold["max_drawdown"]) * 1.5:
            lines.append(f"  - WARNING: forward drawdown > holdout drawdown * 1.5 for {method_key}")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("- `portfolio_equal_weight.csv`")
    lines.append("- `portfolio_vol_weight.csv`")
    lines.append("- `portfolio_corr_min.csv`")
    lines.append("- `correlation_matrix.csv`")
    lines.append("- `portfolio_summary.json`")
    lines.append("")

    report_text = "\n".join(lines).strip() + "\n"
    (run_dir / "portfolio_report.md").write_text(report_text, encoding="utf-8")


def _load_stage1_config(stage1_run_dir: Path) -> ConfigDict:
    with (stage1_run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_feature_data(config: ConfigDict, data_dir: Path) -> dict[str, pd.DataFrame]:
    store = build_data_store(
        backend=str(config.get("data", {}).get("backend", "parquet")),
        data_dir=data_dir,
        base_timeframe=str(config.get("universe", {}).get("base_timeframe") or config["universe"]["timeframe"]),
        resample_source=str(config.get("data", {}).get("resample_source", "direct")),
        partial_last_bucket=bool(config.get("data", {}).get("partial_last_bucket", False)),
    )
    timeframe = str(config["universe"].get("operational_timeframe") or config["universe"]["timeframe"])
    start = config["universe"].get("start")
    end = get_universe_end(config)
    feature_data: dict[str, pd.DataFrame] = {}
    for symbol in config["universe"]["symbols"]:
        frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
        feature_data[str(symbol)] = calculate_features(frame, config=config, symbol=str(symbol), timeframe=timeframe)
    return feature_data


def _resolve_stage1_run_dir(stage1_run_id: str, runs_dir: Path) -> Path:
    if stage1_run_id:
        run_dir = runs_dir / stage1_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Stage-1 run not found: {stage1_run_id}")
        return run_dir

    candidates = sorted(
        [path for path in runs_dir.glob("*_stage1") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No Stage-1 runs found")
    return candidates[0]


def _normalize_method(method: str) -> str:
    value = str(method).strip().lower()
    if value not in _ALLOWED_METHODS:
        raise ValueError(f"Unsupported Stage-2 method: {method}")
    if value in {"corr", "min"}:
        return "corr-min"
    return value


def _select_methods(method: str) -> list[str]:
    if method == "all":
        return ["equal", "vol", "corr-min"]
    return [method]


def _parse_date_range(value: str | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if value is None or value == "n/a" or ".." not in value:
        return None
    start_text, end_text = value.split("..", 1)
    start = pd.Timestamp(start_text)
    end = pd.Timestamp(end_text)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    return start, end


def _clean_range(value: str | None) -> str | None:
    if value is None or value == "n/a":
        return None
    return str(value)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _json_safe_stage2_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    safe["portfolio_methods"] = {
        method_key: {
            key: value
            for key, value in method_payload.items()
            if key not in {"holdout_series", "forward_series"}
        }
        for method_key, method_payload in payload["portfolio_methods"].items()
    }
    return safe
