"""Stage-3.1 Monte Carlo robustness simulation for Stage-2 portfolios."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, validate_config
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.portfolio.builder import (
    INITIAL_PORTFOLIO_CAPITAL,
    Stage1CandidateRecord,
    _load_stage1_config,
    build_candidate_signal_cache,
    load_stage1_candidates,
)
from buffmini.portfolio.walkforward import (
    WindowSpec,
    _compute_input_data_hash,
    _load_raw_data,
    evaluate_candidate_bundles_for_window,
)
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)

DEFAULT_MC_METHODS = ["equal", "vol", "corr-min"]
MC_DD_THRESHOLDS = [0.2, 0.3, 0.4]


@dataclass(frozen=True)
class Stage2MonteCarloContext:
    """Reusable Stage-2 reconstruction context for Monte Carlo evaluation."""

    stage2_run_id: str
    stage2_run_dir: Path
    stage2_summary: dict[str, Any]
    stage1_run_id: str
    stage1_run_dir: Path
    config: dict[str, Any]
    config_hash: str
    data_hash: str
    selected_records: dict[str, Stage1CandidateRecord]
    signal_caches: dict[str, dict[str, pd.DataFrame]]
    initial_capital: float
    round_trip_cost_pct: float
    slippage_pct: float
    holdout_window: WindowSpec
    candidate_bundles: dict[str, dict[str, Any]]


def load_portfolio_trades(
    stage2_run_id: str,
    method: str,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load the Stage-2 holdout portfolio trade stream for one method."""

    context = _load_stage2_context(stage2_run_id=stage2_run_id, runs_dir=runs_dir, data_dir=data_dir)
    return _reconstruct_method_trade_frame(context=context, method=_normalize_method_key(method))


def sample_iid_indices(n_trades: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Sample IID bootstrap trade indices."""

    if int(n_trades) < 1:
        raise ValueError("n_trades must be >= 1")
    if int(n_paths) < 1:
        raise ValueError("n_paths must be >= 1")
    return rng.integers(0, int(n_trades), size=(int(n_paths), int(n_trades)), endpoint=False)


def sample_block_indices(
    n_trades: int,
    n_paths: int,
    block_size_trades: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample block-bootstrap trade indices preserving contiguous trade clusters."""

    if int(n_trades) < 1:
        raise ValueError("n_trades must be >= 1")
    if int(n_paths) < 1:
        raise ValueError("n_paths must be >= 1")
    effective_block = max(1, min(int(block_size_trades), int(n_trades)))
    n_blocks = int(math.ceil(int(n_trades) / effective_block))
    max_start = int(n_trades) - effective_block + 1
    starts = rng.integers(0, max_start, size=(int(n_paths), n_blocks), endpoint=False)
    offsets = np.arange(effective_block, dtype=int)
    indices = np.empty((int(n_paths), n_blocks * effective_block), dtype=int)
    for block_idx in range(n_blocks):
        block = starts[:, [block_idx]] + offsets
        indices[:, block_idx * effective_block : (block_idx + 1) * effective_block] = block
    return indices[:, : int(n_trades)]


def compute_equity_path_metrics(
    trade_pnls: pd.Series | list[float] | np.ndarray,
    initial_equity: float,
    leverage: float = 1.0,
) -> dict[str, float]:
    """Compute deterministic equity-path metrics for one ordered trade sequence."""

    values = np.asarray(pd.Series(trade_pnls, dtype=float).dropna().to_numpy(), dtype=float)
    if values.size == 0:
        return {
            "final_equity": float(initial_equity),
            "total_return_pct": 0.0,
            "max_drawdown": 0.0,
            "worst_trade": 0.0,
            "number_of_trades_used": 0.0,
            "min_equity": float(initial_equity),
        }

    scaled = values * float(leverage)
    cumulative = np.cumsum(scaled, dtype=float)
    equity = float(initial_equity) + cumulative
    equity_with_initial = np.concatenate([[float(initial_equity)], equity])
    peaks = np.maximum.accumulate(equity_with_initial)
    drawdowns = np.where(peaks > 0.0, (peaks - equity_with_initial) / peaks, 0.0)
    max_drawdown = float(np.max(drawdowns))
    final_equity = float(equity[-1])
    total_return_pct = float((final_equity / float(initial_equity)) - 1.0) if float(initial_equity) != 0 else 0.0
    return {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
        "worst_trade": float(np.min(scaled)),
        "number_of_trades_used": float(values.size),
        "min_equity": float(np.min(equity_with_initial)),
    }


def simulate_equity_paths(
    trade_pnls: pd.Series | list[float] | np.ndarray,
    n_paths: int,
    method: str,
    seed: int,
    initial_equity: float,
    leverage: float = 1.0,
    block_size_trades: int = 10,
) -> pd.DataFrame:
    """Simulate Monte Carlo equity paths from an ordered trade PnL stream."""

    values = np.asarray(pd.Series(trade_pnls, dtype=float).dropna().to_numpy(), dtype=float)
    if values.size == 0:
        raise ValueError("trade_pnls must be non-empty")
    if int(n_paths) < 1:
        raise ValueError("n_paths must be >= 1")

    rng = np.random.default_rng(int(seed))
    bootstrap_method = str(method).strip().lower()
    if bootstrap_method == "iid":
        indices = sample_iid_indices(n_trades=values.size, n_paths=int(n_paths), rng=rng)
    elif bootstrap_method == "block":
        indices = sample_block_indices(
            n_trades=values.size,
            n_paths=int(n_paths),
            block_size_trades=int(block_size_trades),
            rng=rng,
        )
    else:
        raise ValueError("method must be 'iid' or 'block'")

    sampled = values[indices] * float(leverage)
    cumulative = np.cumsum(sampled, axis=1, dtype=float)
    equity = float(initial_equity) + cumulative
    initial_column = np.full((int(n_paths), 1), float(initial_equity), dtype=float)
    equity_with_initial = np.concatenate([initial_column, equity], axis=1)
    peaks = np.maximum.accumulate(equity_with_initial, axis=1)
    drawdowns = np.where(peaks > 0.0, (peaks - equity_with_initial) / peaks, 0.0)
    max_drawdown = np.max(drawdowns, axis=1)

    results = pd.DataFrame(
        {
            "final_equity": equity[:, -1],
            "total_return_pct": (equity[:, -1] / float(initial_equity)) - 1.0,
            "max_drawdown": max_drawdown,
            "worst_trade": np.min(sampled, axis=1),
            "number_of_trades_used": np.full(int(n_paths), values.size, dtype=int),
            "min_equity": np.min(equity_with_initial, axis=1),
        }
    )
    return results.astype(
        {
            "final_equity": float,
            "total_return_pct": float,
            "max_drawdown": float,
            "worst_trade": float,
            "number_of_trades_used": int,
            "min_equity": float,
        }
    )


def summarize_mc(
    paths_results: pd.DataFrame,
    initial_equity: float,
    ruin_dd_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate Monte Carlo path results into audit-ready risk statistics."""

    if paths_results.empty:
        raise ValueError("paths_results must be non-empty")

    final_equity = paths_results["final_equity"].astype(float)
    returns = paths_results["total_return_pct"].astype(float)
    max_dd = paths_results["max_drawdown"].astype(float)
    min_equity = paths_results["min_equity"].astype(float)
    ruin_level = float(initial_equity) * (1.0 - float(ruin_dd_threshold))
    tail_losses = returns[returns <= returns.quantile(0.05)]
    cvar5 = float(tail_losses.mean()) if not tail_losses.empty else float(returns.min())

    summary = {
        "final_equity": {
            "median": float(final_equity.quantile(0.50)),
            "p05": float(final_equity.quantile(0.05)),
            "p95": float(final_equity.quantile(0.95)),
        },
        "return_pct": {
            "median": float(returns.quantile(0.50)),
            "p05": float(returns.quantile(0.05)),
            "p95": float(returns.quantile(0.95)),
            "cvar5": cvar5,
        },
        "max_drawdown": {
            "median": float(max_dd.quantile(0.50)),
            "p90": float(max_dd.quantile(0.90)),
            "p95": float(max_dd.quantile(0.95)),
            "p99": float(max_dd.quantile(0.99)),
        },
        "tail_probabilities": {
            "p_return_lt_0": float((returns < 0.0).mean()),
            "p_maxdd_gt_20": float((max_dd > 0.20).mean()),
            "p_maxdd_gt_30": float((max_dd > 0.30).mean()),
            "p_maxdd_gt_40": float((max_dd > 0.40).mean()),
            "p_ruin": float((min_equity <= ruin_level).mean()),
        },
        "ruin_definition": f"equity <= initial_equity * (1 - {float(ruin_dd_threshold):.2f})",
    }
    if not _payload_has_no_inf_nan(summary):
        raise ValueError("Monte Carlo summary contains NaN or infinite values")
    return summary

def run_stage3_monte_carlo(
    stage2_run_id: str,
    methods: list[str] | None = None,
    bootstrap: str = "block",
    block_size_trades: int = 10,
    n_paths: int = 20_000,
    initial_equity: float = 10_000.0,
    ruin_dd_threshold: float = 0.5,
    seed: int = 42,
    leverage: float = 1.0,
    save_paths: bool = False,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    run_id: str | None = None,
    cli_command: str | None = None,
) -> Path:
    """Run Stage-3.1 Monte Carlo robustness simulation from an existing Stage-2 run."""

    bootstrap_method = str(bootstrap).strip().lower()
    if bootstrap_method not in {"iid", "block"}:
        raise ValueError("bootstrap must be 'iid' or 'block'")
    if int(block_size_trades) < 1:
        raise ValueError("block_size_trades must be >= 1")
    if int(n_paths) < 1:
        raise ValueError("n_paths must be >= 1")
    if float(initial_equity) <= 0:
        raise ValueError("initial_equity must be > 0")
    if not 0 < float(ruin_dd_threshold) < 1:
        raise ValueError("ruin_dd_threshold must be between 0 and 1")
    if float(leverage) <= 0:
        raise ValueError("leverage must be > 0")

    context = _load_stage2_context(stage2_run_id=stage2_run_id, runs_dir=runs_dir, data_dir=data_dir)
    requested_methods = [_normalize_method_key(item) for item in (methods or list(DEFAULT_MC_METHODS))]

    method_results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    path_frames: dict[str, pd.DataFrame] = {}

    for offset, method_key in enumerate(requested_methods):
        trades = _reconstruct_method_trade_frame(context=context, method=method_key)
        trade_pnls = trades["pnl"].astype(float)
        paths = simulate_equity_paths(
            trade_pnls=trade_pnls,
            n_paths=int(n_paths),
            method=bootstrap_method,
            seed=int(seed) + offset,
            initial_equity=float(initial_equity),
            leverage=float(leverage),
            block_size_trades=int(block_size_trades),
        )
        summary = summarize_mc(paths_results=paths, initial_equity=float(initial_equity), ruin_dd_threshold=float(ruin_dd_threshold))
        source_stats = _source_trade_stats(trades)
        method_results[method_key] = {
            "trade_count_source": int(len(trades)),
            "source_stats": source_stats,
            "summary": summary,
        }
        summary_rows.append(
            {
                "method": method_key,
                "trade_count_source": int(len(trades)),
                "positive_trade_fraction": float(source_stats["positive_trade_fraction"]),
                "avg_trade_pnl": float(source_stats["avg_trade_pnl"]),
                "trade_pnl_std": float(source_stats["trade_pnl_std"]),
                "return_p05": float(summary["return_pct"]["p05"]),
                "return_median": float(summary["return_pct"]["median"]),
                "return_p95": float(summary["return_pct"]["p95"]),
                "maxdd_p95": float(summary["max_drawdown"]["p95"]),
                "maxdd_p99": float(summary["max_drawdown"]["p99"]),
                "p_return_lt_0": float(summary["tail_probabilities"]["p_return_lt_0"]),
                "p_ruin": float(summary["tail_probabilities"]["p_ruin"]),
            }
        )
        quantile_rows.extend(
            [
                {"method": method_key, "metric": "return_pct", "quantile": "p05", "value": float(summary["return_pct"]["p05"])},
                {"method": method_key, "metric": "return_pct", "quantile": "median", "value": float(summary["return_pct"]["median"])},
                {"method": method_key, "metric": "return_pct", "quantile": "p95", "value": float(summary["return_pct"]["p95"])},
                {"method": method_key, "metric": "max_drawdown", "quantile": "median", "value": float(summary["max_drawdown"]["median"])},
                {"method": method_key, "metric": "max_drawdown", "quantile": "p90", "value": float(summary["max_drawdown"]["p90"])},
                {"method": method_key, "metric": "max_drawdown", "quantile": "p95", "value": float(summary["max_drawdown"]["p95"])},
                {"method": method_key, "metric": "max_drawdown", "quantile": "p99", "value": float(summary["max_drawdown"]["p99"])},
            ]
        )
        tail_rows.append(
            {
                "method": method_key,
                "p_return_lt_0": float(summary["tail_probabilities"]["p_return_lt_0"]),
                "p_maxdd_gt_20": float(summary["tail_probabilities"]["p_maxdd_gt_20"]),
                "p_maxdd_gt_30": float(summary["tail_probabilities"]["p_maxdd_gt_30"]),
                "p_maxdd_gt_40": float(summary["tail_probabilities"]["p_maxdd_gt_40"]),
                "p_ruin": float(summary["tail_probabilities"]["p_ruin"]),
                "cvar5": float(summary["return_pct"]["cvar5"]),
            }
        )
        if save_paths:
            path_frames[method_key] = paths.copy()

    ranking = sorted(
        summary_rows,
        key=lambda row: (float(row["maxdd_p95"]), -float(row["return_p05"])),
    )
    best_method = ranking[0]["method"] if ranking else None
    eligible = bool(
        best_method is not None
        and float(method_results[best_method]["summary"]["tail_probabilities"]["p_ruin"]) < 0.01
        and float(method_results[best_method]["summary"]["max_drawdown"]["p95"]) < 0.30
    )
    recommendation = (
        "Eligible for leverage modeling stage"
        if eligible
        else "Not eligible; improve discovery/exits first"
    )

    settings_payload = {
        "stage2_run_id": context.stage2_run_id,
        "stage1_run_id": context.stage1_run_id,
        "methods": requested_methods,
        "bootstrap": bootstrap_method,
        "block_size_trades": int(block_size_trades),
        "n_paths": int(n_paths),
        "initial_equity": float(initial_equity),
        "ruin_dd_threshold": float(ruin_dd_threshold),
        "seed": int(seed),
        "leverage": float(leverage),
        "save_paths": bool(save_paths),
        "config_hash": context.config_hash,
        "data_hash": context.data_hash,
        "command": cli_command
        or _default_command(
            stage2_run_id=stage2_run_id,
            methods=requested_methods,
            bootstrap=bootstrap_method,
            block_size_trades=int(block_size_trades),
            n_paths=int(n_paths),
            initial_equity=float(initial_equity),
            ruin_dd_threshold=float(ruin_dd_threshold),
            seed=int(seed),
            leverage=float(leverage),
            save_paths=bool(save_paths),
        ),
    }
    summary_payload = {
        "stage2_run_id": context.stage2_run_id,
        "stage1_run_id": context.stage1_run_id,
        "settings": settings_payload,
        "methods": method_results,
        "ranking": ranking,
        "best_method": best_method,
        "recommendation": recommendation,
    }
    if not _payload_has_no_inf_nan(summary_payload):
        raise ValueError("Stage-3.1 summary contains NaN or infinite values")

    summary_hash = stable_hash(summary_payload, length=12)
    resolved_run_id = run_id or f"{utc_now_compact()}_{summary_hash}_stage3_1_mc"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_payload["run_id"] = resolved_run_id

    pd.DataFrame(summary_rows).to_csv(run_dir / "mc_paths_summary.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(run_dir / "mc_quantiles.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(run_dir / "mc_tail_probs.csv", index=False)
    (run_dir / "mc_settings.json").write_text(json.dumps(settings_payload, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "mc_summary.json").write_text(json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8")
    shutil.copyfile(context.stage2_run_dir / "portfolio_summary.json", run_dir / "stage2_metadata.json")
    for method_key, frame in path_frames.items():
        frame.to_parquet(run_dir / f"mc_paths_{method_key.replace('-', '_')}.parquet", index=False)
    _write_mc_report(run_dir=run_dir, summary=summary_payload)

    logger.info("Saved Stage-3.1 Monte Carlo artifacts to %s", run_dir)
    return run_dir


def _load_stage2_context(stage2_run_id: str, runs_dir: Path, data_dir: Path) -> Stage2MonteCarloContext:
    stage2_run_dir = runs_dir / stage2_run_id
    if not stage2_run_dir.exists():
        raise FileNotFoundError(f"Stage-2 run not found: {stage2_run_id}")
    stage2_summary = _load_json(stage2_run_dir / "portfolio_summary.json")
    stage1_run_id = str(stage2_summary["stage1_run_id"])
    stage1_run_dir = runs_dir / stage1_run_id
    if not stage1_run_dir.exists():
        raise FileNotFoundError(f"Stage-1 run not found: {stage1_run_id}")

    config = _load_stage1_config(stage1_run_dir)
    validate_config(config)
    raw_data = _load_raw_data(config=config, data_dir=data_dir)
    feature_data = {symbol: calculate_features(frame) for symbol, frame in raw_data.items()}
    config_hash = compute_config_hash(config)
    data_hash = _compute_input_data_hash(raw_data)

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
        raise ValueError("Stage-3.1 requires Stage-1 candidate artifacts for the Stage-2 selection set")

    signal_caches = {
        candidate_id: build_candidate_signal_cache(candidate=record.to_candidate(), feature_data=feature_data)
        for candidate_id, record in selected_records.items()
    }
    initial_capital = float(INITIAL_PORTFOLIO_CAPITAL * float(config["risk"]["max_concurrent_positions"]))
    round_trip_cost_pct = float(config["costs"]["round_trip_cost_pct"])
    slippage_pct = float(config["costs"]["slippage_pct"])
    holdout_window = _build_holdout_window_from_stage2(stage2_summary)
    candidate_bundles = evaluate_candidate_bundles_for_window(
        window=holdout_window,
        candidate_records=selected_records,
        signal_caches=signal_caches,
        initial_capital=initial_capital,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
    )
    return Stage2MonteCarloContext(
        stage2_run_id=stage2_run_id,
        stage2_run_dir=stage2_run_dir,
        stage2_summary=stage2_summary,
        stage1_run_id=stage1_run_id,
        stage1_run_dir=stage1_run_dir,
        config=config,
        config_hash=config_hash,
        data_hash=data_hash,
        selected_records=selected_records,
        signal_caches=signal_caches,
        initial_capital=initial_capital,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        holdout_window=holdout_window,
        candidate_bundles=candidate_bundles,
    )

def _build_holdout_window_from_stage2(stage2_summary: dict[str, Any]) -> WindowSpec:
    available_methods = [payload for payload in stage2_summary["portfolio_methods"].values() if "holdout" in payload]
    if not available_methods:
        raise ValueError("Stage-2 summary does not contain holdout metrics")
    holdout_range = str(available_methods[0]["holdout"]["date_range"])
    start_text, end_text = holdout_range.split("..", 1)
    start = _ensure_utc(start_text)
    end = _ensure_utc(end_text)
    bar_count = int(((end - start) / pd.Timedelta(hours=1)) + 1)
    return WindowSpec(
        name="Holdout",
        kind="holdout",
        expected_start=start,
        expected_end=end,
        actual_start=start,
        actual_end=end,
        truncated=False,
        enough_data=bool(bar_count >= 2),
        bar_count=bar_count,
        note="Stage-2 holdout trade source window.",
    )


def _reconstruct_method_trade_frame(context: Stage2MonteCarloContext, method: str) -> pd.DataFrame:
    payload = context.stage2_summary["portfolio_methods"].get(method)
    if payload is None:
        raise ValueError(f"Stage-2 summary does not contain method: {method}")
    weights = {str(candidate_id): float(weight) for candidate_id, weight in payload.get("weights", {}).items()}
    frames: list[pd.DataFrame] = []
    for candidate_id, bundle in context.candidate_bundles.items():
        trades = bundle.get("trades", pd.DataFrame())
        weight = float(weights.get(candidate_id, 0.0))
        if weight <= 0.0 or trades.empty:
            continue
        scaled = trades.copy()
        scaled["entry_ts"] = pd.to_datetime(scaled["entry_time"], utc=True)
        scaled["exit_ts"] = pd.to_datetime(scaled["exit_time"], utc=True)
        scaled["pnl"] = scaled["candidate_pnl"].astype(float) * weight
        scaled["pnl_pct"] = scaled["candidate_return_pct"].astype(float) * weight
        scaled["weighted_candidate_id"] = candidate_id
        frames.append(
            scaled[
                [
                    "entry_ts",
                    "exit_ts",
                    "pnl",
                    "pnl_pct",
                    "side",
                    "symbol",
                    "weighted_candidate_id",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(columns=["entry_ts", "exit_ts", "pnl", "pnl_pct", "side", "symbol", "weighted_candidate_id"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["exit_ts", "entry_ts", "weighted_candidate_id", "symbol"]).reset_index(drop=True)
    return combined


def _source_trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "positive_trade_fraction": 0.0,
            "avg_trade_pnl": 0.0,
            "trade_pnl_std": 0.0,
        }
    pnls = trades["pnl"].astype(float)
    return {
        "positive_trade_fraction": float((pnls > 0.0).mean()),
        "avg_trade_pnl": float(pnls.mean()),
        "trade_pnl_std": float(pnls.std(ddof=0)),
    }


def _write_mc_report(run_dir: Path, summary: dict[str, Any]) -> None:
    settings = summary["settings"]
    lines: list[str] = []
    lines.append("# Stage-3.1 Monte Carlo Robustness Report")
    lines.append("")
    lines.append("## Section 1 - Provenance")
    lines.append(f"- Stage-2 run_id used: `{summary['stage2_run_id']}`")
    lines.append(f"- Stage-1 run_id referenced: `{summary['stage1_run_id']}`")
    lines.append(f"- exact CLI command: `{settings['command']}`")
    lines.append(f"- seed: `{settings['seed']}`")
    lines.append(f"- n_paths: `{settings['n_paths']}`")
    lines.append(f"- bootstrap type: `{settings['bootstrap']}`")
    lines.append(f"- block_size_trades: `{settings['block_size_trades']}`")
    lines.append(f"- initial_equity: `{settings['initial_equity']}`")
    lines.append(f"- leverage: `{settings['leverage']}`")
    lines.append(f"- ruin_dd_threshold: `{settings['ruin_dd_threshold']}`")
    lines.append(f"- config hash: `{settings['config_hash']}`")
    lines.append(f"- data hash: `{settings['data_hash']}`")
    for method_key, payload in summary["methods"].items():
        lines.append(f"- trade_count_source[{method_key}]: `{payload['trade_count_source']}`")
    lines.append("")

    lines.append("## Section 2 - Method sanity checks")
    for method_key, payload in summary["methods"].items():
        source = payload["source_stats"]
        lines.append(f"### {method_key}")
        lines.append(f"- trade_count_source: `{payload['trade_count_source']}`")
        lines.append(f"- fraction_positive_trades: `{source['positive_trade_fraction']:.4f}`")
        lines.append(f"- avg_trade_pnl: `{source['avg_trade_pnl']:.4f}`")
        lines.append(f"- trade_pnl_std: `{source['trade_pnl_std']:.4f}`")
        lines.append("- block bootstrap note: contiguous trade blocks are resampled to preserve clustering and local regime dependence.")
        lines.append("")

    lines.append("## Section 3 - Monte Carlo Results")
    lines.append("| method | return median | return p05 | return p95 | maxDD median | maxDD p90 | maxDD p95 | maxDD p99 | P(return<0) | P(maxDD>20%) | P(maxDD>30%) | P(maxDD>40%) | P(ruin) | CVaR5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method_key in DEFAULT_MC_METHODS:
        payload = summary["methods"].get(method_key)
        if payload is None:
            continue
        mc = payload["summary"]
        lines.append(
            f"| {method_key} | {mc['return_pct']['median']:.4f} | {mc['return_pct']['p05']:.4f} | {mc['return_pct']['p95']:.4f} | "
            f"{mc['max_drawdown']['median']:.4f} | {mc['max_drawdown']['p90']:.4f} | {mc['max_drawdown']['p95']:.4f} | {mc['max_drawdown']['p99']:.4f} | "
            f"{mc['tail_probabilities']['p_return_lt_0']:.4f} | {mc['tail_probabilities']['p_maxdd_gt_20']:.4f} | {mc['tail_probabilities']['p_maxdd_gt_30']:.4f} | "
            f"{mc['tail_probabilities']['p_maxdd_gt_40']:.4f} | {mc['tail_probabilities']['p_ruin']:.4f} | {mc['return_pct']['cvar5']:.4f} |"
        )
    lines.append("")

    lines.append("## Section 4 - Comparison + Recommendation")
    lines.append("- Ranking criterion: lowest p95 max drawdown first, then highest p05 return.")
    for rank, row in enumerate(summary["ranking"], start=1):
        lines.append(
            f"- {rank}. {row['method']}: p95_maxDD={float(row['maxdd_p95']):.4f}, p05_return={float(row['return_p05']):.4f}, P(ruin)={float(row['p_ruin']):.4f}"
        )
    best_method = summary.get("best_method")
    if best_method is not None:
        best = summary["methods"][best_method]["summary"]
        lines.append(
            f"- Best method `{best_method}`: P(ruin)={float(best['tail_probabilities']['p_ruin']):.4f}, p95_maxDD={float(best['max_drawdown']['p95']):.4f}"
        )
    lines.append(f"- Recommendation: {summary['recommendation']}")
    lines.append("")

    lines.append("## Section 5 - Integrity")
    lines.append("- Determinism: covered by tests/test_stage3_monte_carlo.py using repeated same-seed simulation and summary comparison.")
    lines.append(f"- no NaN/inf in summaries: `{_payload_has_no_inf_nan(summary)}`")
    lines.append(f"- ruin definition: `{next(iter(summary['methods'].values()))['summary']['ruin_definition'] if summary['methods'] else 'n/a'}`")

    (run_dir / "mc_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _normalize_method_key(method: str) -> str:
    value = str(method).strip().lower()
    if value not in {"equal", "vol", "corr-min", "corr", "min"}:
        raise ValueError(f"Unsupported method: {method}")
    if value in {"corr", "min"}:
        return "corr-min"
    return value


def _default_command(
    stage2_run_id: str,
    methods: list[str],
    bootstrap: str,
    block_size_trades: int,
    n_paths: int,
    initial_equity: float,
    ruin_dd_threshold: float,
    seed: int,
    leverage: float,
    save_paths: bool,
) -> str:
    command = (
        "python scripts/run_stage3_monte_carlo.py "
        f"--stage2-run-id {stage2_run_id} --methods {','.join(methods)} --bootstrap {bootstrap} "
        f"--block-size-trades {block_size_trades} --n-paths {n_paths} --initial-equity {initial_equity} "
        f"--ruin-dd-threshold {ruin_dd_threshold} --seed {seed} --leverage {leverage}"
    )
    if save_paths:
        command += " --save-paths"
    return command


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_has_no_inf_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_payload_has_no_inf_nan(item) for item in value.values())
    if isinstance(value, list):
        return all(_payload_has_no_inf_nan(item) for item in value)
    return True
