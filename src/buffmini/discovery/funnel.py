"""Stage-1 random-search optimization funnel."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals
from buffmini.config import compute_config_hash
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.storage import load_parquet
from buffmini.data.window import slice_last_n_months
from buffmini.discovery.generator import (
    Candidate,
    candidate_to_strategy_spec,
    complexity_penalty,
    perturb_candidate,
    sample_candidate,
)
from buffmini.types import ConfigDict
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)


@dataclass
class CandidateEval:
    """Candidate evaluation output."""

    candidate: Candidate
    stage: str
    score: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    trade_count: float
    final_equity: float
    return_pct: float
    complexity_penalty: float
    instability_penalty: float
    date_range: str

    def to_row(self, rank: int | None = None) -> dict[str, Any]:
        row = {
            "stage": self.stage,
            "candidate_id": self.candidate.candidate_id,
            "family": self.candidate.family,
            "gating_mode": self.candidate.gating_mode,
            "exit_mode": self.candidate.exit_mode,
            "score": self.score,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "final_equity": self.final_equity,
            "return_pct": self.return_pct,
            "complexity_penalty": self.complexity_penalty,
            "instability_penalty": self.instability_penalty,
            "date_range": self.date_range,
            "params": json.dumps(self.candidate.params, sort_keys=True),
        }
        if rank is not None:
            row["rank"] = int(rank)
        return row


def run_stage1_optimization(
    config: ConfigDict,
    config_path: Path,
    dry_run: bool = False,
    run_id: str | None = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    candidate_count: int | None = None,
    seed: int | None = None,
    cost_pct: float | None = None,
    stage_a_months: int | None = None,
    stage_b_months: int | None = None,
    holdout_months: int | None = None,
    docs_report_path: Path | None = None,
) -> Path:
    """Run Stage-1 random-search funnel and save artifacts."""

    started_at = time.time()

    stage1 = config["evaluation"]["stage1"]
    search_space = stage1["search_space"]
    weights = stage1["weights"]

    resolved_seed = int(seed if seed is not None else config["search"]["seed"])
    resolved_candidate_count = int(candidate_count if candidate_count is not None else stage1["candidate_count"])
    resolved_top_k = int(stage1["top_k"])
    resolved_top_m = int(stage1["top_m"])
    resolved_stage_a_months = int(stage_a_months if stage_a_months is not None else stage1["stage_a_months"])
    resolved_stage_b_months = int(stage_b_months if stage_b_months is not None else stage1["stage_b_months"])
    resolved_holdout_months = int(holdout_months if holdout_months is not None else stage1["holdout_months"])
    walkforward_splits = int(stage1["walkforward_splits"])
    early_stop_patience = int(stage1["early_stop_patience"])
    min_stage_a_evals = int(stage1["min_stage_a_evals"])

    costs = config["costs"]
    round_trip_cost_pct = float(cost_pct if cost_pct is not None else costs["round_trip_cost_pct"])
    slippage_pct = float(costs["slippage_pct"])

    risk = config["risk"]
    initial_capital = 10_000.0 * float(risk["max_concurrent_positions"])

    universe_symbols = list(config["universe"]["symbols"])
    required_symbols = [s for s in ["BTC/USDT", "ETH/USDT"] if s in universe_symbols]
    if len(required_symbols) < 2:
        raise ValueError("Stage-1 requires BTC/USDT and ETH/USDT in config.universe.symbols")

    raw_data = _load_stage1_data(
        symbols=required_symbols,
        timeframe=config["universe"]["timeframe"],
        data_dir=data_dir,
        dry_run=dry_run,
        start=config["universe"]["start"],
        seed=resolved_seed,
    )
    feature_data = {symbol: calculate_features(frame) for symbol, frame in raw_data.items()}
    data_hash = _compute_data_hash(feature_data)

    config_hash = compute_config_hash(config)
    resolved_run_id = run_id or f"{utc_now_compact()}_{config_hash}_stage1"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    search_meta = {
        "seed": resolved_seed,
        "candidate_count": resolved_candidate_count,
        "top_k": resolved_top_k,
        "top_m": resolved_top_m,
        "stage_a_months": resolved_stage_a_months,
        "stage_b_months": resolved_stage_b_months,
        "holdout_months": resolved_holdout_months,
        "walkforward_splits": walkforward_splits,
        "early_stop_patience": early_stop_patience,
        "min_stage_a_evals": min_stage_a_evals,
        "round_trip_cost_pct": round_trip_cost_pct,
        "slippage_pct": slippage_pct,
        "weights": weights,
        "search_space": search_space,
        "data_hash": data_hash,
        "dry_run": dry_run,
        "config_path": str(config_path),
    }
    with (run_dir / "search_space.json").open("w", encoding="utf-8") as handle:
        json.dump(search_meta, handle, indent=2)

    stage_a_data = {
        "BTC/USDT": _slice_by_months(feature_data["BTC/USDT"], resolved_stage_a_months),
    }
    stage_b_data = {
        symbol: _slice_by_months(feature_data[symbol], resolved_stage_b_months)
        for symbol in required_symbols
    }
    stage_c_data = {
        symbol: frame.copy().sort_values("timestamp").reset_index(drop=True)
        for symbol, frame in feature_data.items()
    }

    rng = np.random.default_rng(resolved_seed)
    leaderboard_rows: list[dict[str, Any]] = []

    # Stage A: random search on BTC, short window
    stage_a_results: list[CandidateEval] = []
    best_score = -math.inf
    no_improve = 0

    for idx in range(resolved_candidate_count):
        candidate = sample_candidate(index=idx + 1, rng=rng, search_space=search_space)
        metrics = _evaluate_candidate_metrics(
            candidate=candidate,
            data_by_symbol=stage_a_data,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )

        instability = 0.0
        score = _compute_score(
            metrics=metrics,
            candidate=candidate,
            weights=weights,
            instability_penalty=instability,
        )
        result = CandidateEval(
            candidate=candidate,
            stage="A",
            score=score,
            expectancy=metrics["expectancy"],
            profit_factor=metrics["profit_factor"],
            max_drawdown=metrics["max_drawdown"],
            trade_count=metrics["trade_count"],
            final_equity=metrics["final_equity"],
            return_pct=metrics["return_pct"],
            complexity_penalty=complexity_penalty(candidate),
            instability_penalty=instability,
            date_range=metrics["date_range"],
        )
        stage_a_results.append(result)

        if score > best_score:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1

        if (idx + 1) >= min_stage_a_evals and no_improve >= early_stop_patience:
            logger.info("Stage A early stop at %s candidates", idx + 1)
            break

    stage_a_sorted = sorted(stage_a_results, key=lambda x: x.score, reverse=True)
    top_k_candidates = [row.candidate for row in stage_a_sorted[:resolved_top_k]]

    for rank, item in enumerate(stage_a_sorted[:resolved_top_k], start=1):
        leaderboard_rows.append(item.to_row(rank=rank))

    # Stage B: medium data, top K re-evaluation with instability penalty
    stage_b_results: list[CandidateEval] = []
    instability_symbol = {"BTC/USDT": stage_b_data["BTC/USDT"]}

    for candidate in top_k_candidates:
        metrics = _evaluate_candidate_metrics(
            candidate=candidate,
            data_by_symbol=stage_b_data,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )

        instability = _instability_penalty(
            candidate=candidate,
            base_metrics=metrics,
            data_by_symbol=instability_symbol,
            search_space=search_space,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
            weights=weights,
        )
        score = _compute_score(metrics=metrics, candidate=candidate, weights=weights, instability_penalty=instability)

        stage_b_results.append(
            CandidateEval(
                candidate=candidate,
                stage="B",
                score=score,
                expectancy=metrics["expectancy"],
                profit_factor=metrics["profit_factor"],
                max_drawdown=metrics["max_drawdown"],
                trade_count=metrics["trade_count"],
                final_equity=metrics["final_equity"],
                return_pct=metrics["return_pct"],
                complexity_penalty=complexity_penalty(candidate),
                instability_penalty=instability,
                date_range=metrics["date_range"],
            )
        )

    stage_b_sorted = sorted(stage_b_results, key=lambda x: x.score, reverse=True)
    top_m_candidates = [row.candidate for row in stage_b_sorted[:resolved_top_m]]

    for rank, item in enumerate(stage_b_sorted[:resolved_top_m], start=1):
        leaderboard_rows.append(item.to_row(rank=rank))

    # Stage C: walk-forward + holdout on top M
    stage_c_results: list[CandidateEval] = []
    for candidate in top_m_candidates:
        holdout_metrics, wf_std_penalty = _evaluate_stage_c_candidate(
            candidate=candidate,
            data_by_symbol=stage_c_data,
            holdout_months=resolved_holdout_months,
            walkforward_splits=walkforward_splits,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )

        perturb_penalty = _instability_penalty(
            candidate=candidate,
            base_metrics=holdout_metrics,
            data_by_symbol={"BTC/USDT": _slice_by_months(stage_c_data["BTC/USDT"], resolved_holdout_months)},
            search_space=search_space,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
            weights=weights,
        )
        instability = perturb_penalty + wf_std_penalty

        score = _compute_score(
            metrics=holdout_metrics,
            candidate=candidate,
            weights=weights,
            instability_penalty=instability,
        )

        stage_c_results.append(
            CandidateEval(
                candidate=candidate,
                stage="C",
                score=score,
                expectancy=holdout_metrics["expectancy"],
                profit_factor=holdout_metrics["profit_factor"],
                max_drawdown=holdout_metrics["max_drawdown"],
                trade_count=holdout_metrics["trade_count"],
                final_equity=holdout_metrics["final_equity"],
                return_pct=holdout_metrics["return_pct"],
                complexity_penalty=complexity_penalty(candidate),
                instability_penalty=instability,
                date_range=holdout_metrics["date_range"],
            )
        )

    stage_c_sorted = sorted(stage_c_results, key=lambda x: x.score, reverse=True)
    top_three = stage_c_sorted[:3]

    for rank, item in enumerate(stage_c_sorted, start=1):
        leaderboard_rows.append(item.to_row(rank=rank))

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    strategies_payload: list[dict[str, Any]] = []
    for rank, item in enumerate(top_three, start=1):
        spec = candidate_to_strategy_spec(item.candidate)
        strategies_payload.append(
            {
                "rank": rank,
                "candidate_id": item.candidate.candidate_id,
                "family": item.candidate.family,
                "strategy_name": spec.name,
                "gating_mode": item.candidate.gating_mode,
                "exit_mode": item.candidate.exit_mode,
                "rules": {
                    "entry": spec.entry_rules,
                    "exit": spec.exit_rules,
                },
                "parameters": item.candidate.params,
                "metrics_holdout": {
                    "score": item.score,
                    "profit_factor": item.profit_factor,
                    "expectancy": item.expectancy,
                    "max_drawdown": item.max_drawdown,
                    "trade_count": item.trade_count,
                    "final_equity": item.final_equity,
                    "return_pct": item.return_pct,
                    "date_range": item.date_range,
                },
            }
        )

    with (run_dir / "strategies.json").open("w", encoding="utf-8") as handle:
        json.dump(strategies_payload, handle, indent=2)

    duration_sec = time.time() - started_at
    summary = {
        "run_id": resolved_run_id,
        "stage_version": "stage1",
        "seed": resolved_seed,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "candidate_count_stage_a": len(stage_a_results),
        "candidate_count_stage_b": len(stage_b_results),
        "candidate_count_stage_c": len(stage_c_results),
        "top_k": resolved_top_k,
        "top_m": resolved_top_m,
        "top_n": len(top_three),
        "round_trip_cost_pct": round_trip_cost_pct,
        "stage_a_months": resolved_stage_a_months,
        "stage_b_months": resolved_stage_b_months,
        "holdout_months": resolved_holdout_months,
        "walkforward_splits": walkforward_splits,
        "runtime_seconds": duration_sec,
        "dry_run": dry_run,
        "best": strategies_payload[0] if strategies_payload else None,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _write_stage1_report(
        run_dir=run_dir,
        summary=summary,
        top_three=strategies_payload,
        docs_report_path=docs_report_path,
    )

    if top_three:
        best = top_three[0]
        logger.info(
            "Stage-1 best candidate: %s | PF=%.4f | expectancy=%.4f | max_dd=%.4f",
            best.candidate.candidate_id,
            best.profit_factor,
            best.expectancy,
            best.max_drawdown,
        )
    logger.info("Saved Stage-1 artifacts to %s", run_dir)
    return run_dir


def _evaluate_stage_c_candidate(
    candidate: Candidate,
    data_by_symbol: dict[str, pd.DataFrame],
    holdout_months: int,
    walkforward_splits: int,
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
) -> tuple[dict[str, float | str], float]:
    holdout_data: dict[str, pd.DataFrame] = {}
    train_data: dict[str, pd.DataFrame] = {}

    for symbol, data in data_by_symbol.items():
        data_sorted = data.sort_values("timestamp").reset_index(drop=True)
        holdout, _ = slice_last_n_months(data_sorted, window_months=holdout_months, end_mode="latest")
        holdout_start = pd.to_datetime(holdout["timestamp"], utc=True).iloc[0]

        train = data_sorted[pd.to_datetime(data_sorted["timestamp"], utc=True) < holdout_start].reset_index(drop=True)
        if train.empty:
            train = data_sorted.iloc[:-len(holdout)].reset_index(drop=True) if len(data_sorted) > len(holdout) else data_sorted.copy()

        holdout_data[symbol] = holdout
        train_data[symbol] = train

    holdout_metrics = _evaluate_candidate_metrics(
        candidate=candidate,
        data_by_symbol=holdout_data,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        initial_capital=initial_capital,
    )

    wf_scores: list[float] = []
    for split_idx in range(walkforward_splits):
        split_data: dict[str, pd.DataFrame] = {}
        for symbol, train in train_data.items():
            if train.empty:
                continue
            split = _rolling_split_test_window(train, split_idx=split_idx, total_splits=walkforward_splits)
            if split is None:
                continue
            split_data[symbol] = split

        if not split_data:
            continue

        split_metrics = _evaluate_candidate_metrics(
            candidate=candidate,
            data_by_symbol=split_data,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        raw = _raw_score(
            expectancy=float(split_metrics["expectancy"]),
            profit_factor=float(split_metrics["profit_factor"]),
            max_drawdown=float(split_metrics["max_drawdown"]),
            complexity=complexity_penalty(candidate),
            instability=0.0,
            weights={
                "expectancy": 1.0,
                "log_profit_factor": 1.0,
                "max_drawdown": 1.0,
                "complexity": 1.0,
                "instability": 1.0,
            },
        )
        wf_scores.append(raw)

    wf_std_penalty = float(np.std(wf_scores)) if wf_scores else 0.0
    return holdout_metrics, wf_std_penalty


def _rolling_split_test_window(train: pd.DataFrame, split_idx: int, total_splits: int) -> pd.DataFrame | None:
    if train.empty:
        return None

    n = len(train)
    test_len = max(120, n // (total_splits + 2))
    test_start = n - (total_splits - split_idx) * test_len
    test_end = min(n, test_start + test_len)

    if test_start < 0 or test_start >= n or test_end <= test_start:
        return None

    window = train.iloc[test_start:test_end].reset_index(drop=True)
    return window if not window.empty else None


def _evaluate_candidate_metrics(
    candidate: Candidate,
    data_by_symbol: dict[str, pd.DataFrame],
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
) -> dict[str, float | str]:
    spec = candidate_to_strategy_spec(candidate)

    expectancy_list: list[float] = []
    pf_list: list[float] = []
    dd_list: list[float] = []
    trade_count_total = 0.0
    final_equity_list: list[float] = []
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None

    for symbol, frame in data_by_symbol.items():
        if frame.empty:
            continue

        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        current_min = timestamps.iloc[0]
        current_max = timestamps.iloc[-1]
        min_ts = current_min if min_ts is None else min(min_ts, current_min)
        max_ts = current_max if max_ts is None else max(max_ts, current_max)

        eval_frame = frame.copy()
        eval_frame["signal"] = generate_signals(eval_frame, spec, gating_mode=candidate.gating_mode)

        result = run_backtest(
            frame=eval_frame,
            strategy_name=spec.name,
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

        expectancy_list.append(float(result.metrics["expectancy"]))
        pf_list.append(float(result.metrics["profit_factor"]))
        dd_list.append(float(result.metrics["max_drawdown"]))
        trade_count_total += float(result.metrics["trade_count"])

        if not result.equity_curve.empty:
            final_equity_list.append(float(result.equity_curve["equity"].iloc[-1]))

    if not expectancy_list:
        return {
            "expectancy": -1e9,
            "profit_factor": 0.0,
            "max_drawdown": 1.0,
            "trade_count": 0.0,
            "final_equity": 0.0,
            "return_pct": -1.0,
            "date_range": "n/a",
        }

    final_equity = float(np.mean(final_equity_list)) if final_equity_list else 0.0
    return_pct = (final_equity - float(initial_capital)) / float(initial_capital)
    date_range = f"{min_ts.isoformat()}..{max_ts.isoformat()}" if min_ts is not None and max_ts is not None else "n/a"

    return {
        "expectancy": float(np.mean(expectancy_list)),
        "profit_factor": float(np.mean(pf_list)),
        "max_drawdown": float(np.mean(dd_list)),
        "trade_count": float(trade_count_total),
        "final_equity": final_equity,
        "return_pct": float(return_pct),
        "date_range": date_range,
    }


def _compute_score(
    metrics: dict[str, float | str],
    candidate: Candidate,
    weights: dict[str, float],
    instability_penalty: float,
) -> float:
    return _raw_score(
        expectancy=float(metrics["expectancy"]),
        profit_factor=float(metrics["profit_factor"]),
        max_drawdown=float(metrics["max_drawdown"]),
        complexity=complexity_penalty(candidate),
        instability=float(instability_penalty),
        weights=weights,
    )


def _raw_score(
    expectancy: float,
    profit_factor: float,
    max_drawdown: float,
    complexity: float,
    instability: float,
    weights: dict[str, float],
) -> float:
    exp = float(expectancy)
    if not math.isfinite(exp):
        exp = -1_000.0

    pf = float(profit_factor)
    if not math.isfinite(pf):
        pf = 10.0
    pf = min(max(pf, 1e-6), 10.0)
    pf_log = math.log(pf)

    dd = float(max_drawdown)
    if not math.isfinite(dd):
        dd = 1.0
    dd = min(max(dd, 0.0), 1.0)

    comp = float(complexity)
    if not math.isfinite(comp):
        comp = 1.0

    instab = float(instability)
    if not math.isfinite(instab):
        instab = 1.0

    return (
        float(weights["expectancy"]) * exp
        + float(weights["log_profit_factor"]) * pf_log
        - float(weights["max_drawdown"]) * dd
        - float(weights["complexity"]) * comp
        - float(weights["instability"]) * instab
    )


def _instability_penalty(
    candidate: Candidate,
    base_metrics: dict[str, float | str],
    data_by_symbol: dict[str, pd.DataFrame],
    search_space: dict[str, Any],
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
    weights: dict[str, float],
) -> float:
    base_score = _compute_score(metrics=base_metrics, candidate=candidate, weights=weights, instability_penalty=0.0)

    perturbed = perturb_candidate(candidate=candidate, search_space=search_space, pct=0.1)
    if not perturbed:
        return 0.0

    perturb_scores: list[float] = []
    for variant in perturbed:
        metrics = _evaluate_candidate_metrics(
            candidate=variant,
            data_by_symbol=data_by_symbol,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        perturb_scores.append(_compute_score(metrics=metrics, candidate=variant, weights=weights, instability_penalty=0.0))

    mean_perturb = float(np.mean(perturb_scores)) if perturb_scores else base_score
    collapse = max(0.0, base_score - mean_perturb)
    return collapse


def _slice_by_months(frame: pd.DataFrame, months: int) -> pd.DataFrame:
    sliced, _ = slice_last_n_months(frame, window_months=int(months), end_mode="latest")
    return sliced


def _compute_data_hash(data_by_symbol: dict[str, pd.DataFrame]) -> str:
    payload: dict[str, Any] = {}
    for symbol, frame in sorted(data_by_symbol.items(), key=lambda x: x[0]):
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        payload[symbol] = {
            "rows": int(len(frame)),
            "start": timestamps.iloc[0].isoformat() if not frame.empty else None,
            "end": timestamps.iloc[-1].isoformat() if not frame.empty else None,
            "close_mean": float(frame["close"].astype(float).mean()) if not frame.empty else None,
            "close_std": float(frame["close"].astype(float).std(ddof=0)) if not frame.empty else None,
        }
    return stable_hash(payload, length=16)


def _load_stage1_data(
    symbols: list[str],
    timeframe: str,
    data_dir: Path,
    dry_run: bool,
    start: str | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    if dry_run:
        for symbol in symbols:
            loaded[symbol] = _generate_synthetic_ohlcv(symbol=symbol, start=start, bars=2400, seed=seed)
        return loaded

    for symbol in symbols:
        loaded[symbol] = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    return loaded


def _generate_synthetic_ohlcv(symbol: str, start: str | None, bars: int, seed: int) -> pd.DataFrame:
    bars = max(int(bars), 400)
    symbol_seed = int.from_bytes(stable_hash(f"{seed}:{symbol}", length=16).encode("utf-8")[:8], "big", signed=False)
    rng = np.random.default_rng(symbol_seed)

    start_ts = pd.Timestamp(start, tz="UTC") if start else pd.Timestamp("2020-01-01T00:00:00Z")
    timestamps = pd.date_range(start=start_ts, periods=bars, freq="h", tz="UTC")

    base_price = 100.0 + (symbol_seed % 40)
    log_returns = rng.normal(loc=0.0001, scale=0.006, size=bars)
    close = base_price * np.exp(np.cumsum(log_returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    spread = rng.uniform(0.0005, 0.008, size=bars)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.uniform(500, 5000, size=bars)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _write_stage1_report(
    run_dir: Path,
    summary: dict[str, Any],
    top_three: list[dict[str, Any]],
    docs_report_path: Path | None,
) -> None:
    report_lines: list[str] = []
    report_lines.append("# Stage-1 Auto Optimization Report")
    report_lines.append("")
    report_lines.append(f"- run_id: `{summary['run_id']}`")
    report_lines.append(f"- stage: `{summary['stage_version']}`")
    report_lines.append(f"- runtime_seconds: `{summary['runtime_seconds']:.2f}`")
    report_lines.append(f"- seed: `{summary['seed']}`")
    report_lines.append(f"- config_hash: `{summary['config_hash']}`")
    report_lines.append(f"- data_hash: `{summary['data_hash']}`")
    report_lines.append(f"- cost(round_trip_cost_pct): `{summary['round_trip_cost_pct']}`")
    report_lines.append(f"- candidates A/B/C: `{summary['candidate_count_stage_a']}/{summary['candidate_count_stage_b']}/{summary['candidate_count_stage_c']}`")
    report_lines.append("")

    report_lines.append("## Top 3 Candidates")
    if not top_three:
        report_lines.append("No candidates produced.")
    else:
        for item in top_three:
            metrics = item["metrics_holdout"]
            report_lines.append(f"### Rank {item['rank']} - {item['family']}")
            report_lines.append(f"- Strategy: `{item['strategy_name']}`")
            report_lines.append(f"- Gating: `{item['gating_mode']}`")
            report_lines.append(f"- Exit mode: `{item['exit_mode']}`")
            report_lines.append(f"- Entry rules: {item['rules']['entry']}")
            report_lines.append(f"- Exit rules: {item['rules']['exit']}")
            report_lines.append(f"- Parameters: `{json.dumps(item['parameters'], sort_keys=True)}`")
            report_lines.append(
                "- Holdout metrics: "
                f"trade_count={metrics['trade_count']:.0f}, "
                f"PF={metrics['profit_factor']:.4f}, "
                f"expectancy={metrics['expectancy']:.4f}, "
                f"max_dd={metrics['max_drawdown']:.4f}, "
                f"return_pct={metrics['return_pct']:.4f}"
            )
            report_lines.append(f"- Holdout range: `{metrics['date_range']}`")
            report_lines.append("")

    # Persist under run artifacts and docs root.
    report_text = "\n".join(report_lines).strip() + "\n"
    (run_dir / "stage1_report.md").write_text(report_text, encoding="utf-8")

    resolved_docs_path = docs_report_path or (Path("docs") / "stage1_auto_optimization_report.md")
    resolved_docs_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_docs_path.write_text(report_text, encoding="utf-8")
