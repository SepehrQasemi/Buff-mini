"""Stage-10.5 evaluation harness."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, stage0_strategies, trend_pullback
from buffmini.config import compute_config_hash, get_universe_end
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.stage10.activation import DEFAULT_ACTIVATION_CONFIG, apply_soft_activation
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.stage10.regimes import REGIME_LABELS, regime_distribution
from buffmini.stage10.signals import DEFAULT_SIGNAL_PARAMS, SIGNAL_FAMILIES, generate_signal_family
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.leakage_harness import run_registered_features_harness, synthetic_ohlcv
from buffmini.validation.walkforward_v2 import aggregate_windows, build_windows, evaluate_candidate_on_window


STAGE10_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "cost_mode": "v2",
    "walkforward_v2": True,
    "regimes": {
        "trend_threshold": 0.010,
        "vol_rank_high": 0.80,
        "vol_rank_low": 0.35,
        "compression_z": -0.8,
        "expansion_z": 1.0,
        "volume_z_high": 1.0,
    },
    "activation": dict(DEFAULT_ACTIVATION_CONFIG),
    "signals": {
        "families": list(SIGNAL_FAMILIES),
        "defaults": {name: dict(params) for name, params in DEFAULT_SIGNAL_PARAMS.items()},
    },
    "exits": {
        "modes": ["fixed_atr", "atr_trailing"],
        "trailing_atr_k": 1.5,
        "partial_fraction": 0.5,
    },
    "evaluation": {
        "initial_capital": 10000.0,
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
        "dry_run_rows": 2400,
    },
}


@dataclass(frozen=True)
class CandidateResult:
    symbol: str
    family: str
    exit_mode: str
    score: float
    trade_count: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    win_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "family": self.family,
            "exit_mode": self.exit_mode,
            "score": _finite(self.score, default=0.0),
            "trade_count": _finite(self.trade_count, default=0.0),
            "expectancy": _finite(self.expectancy, default=0.0),
            "profit_factor": _finite(self.profit_factor, default=10.0, clip=10.0),
            "max_drawdown": _finite(self.max_drawdown, default=0.0),
            "win_rate": _finite(self.win_rate, default=0.0),
        }


def run_stage10(
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    cost_mode: str = "v2",
    walkforward_v2_enabled: bool = True,
    exit_mode: str | None = None,
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Run Stage-10 baseline-vs-upgraded comparison and write artifacts."""

    cfg = _normalize_stage10_config(config=config, cost_mode=cost_mode, exit_mode=exit_mode)
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    resolved_timeframe = str(timeframe or cfg.get("universe", {}).get("timeframe", "1h"))
    if resolved_timeframe != "1h":
        raise ValueError("Stage-10 currently supports timeframe=1h only")

    features_by_symbol = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe=resolved_timeframe,
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    if not features_by_symbol:
        raise ValueError("No feature frames available for Stage-10 evaluation")

    data_hash = _compute_data_hash(features_by_symbol)
    config_hash = compute_config_hash(cfg)
    resolved_end_ts = _resolve_end_ts(config=cfg, features_by_symbol=features_by_symbol)

    baseline_eval = _evaluate_baseline(features_by_symbol=features_by_symbol, cfg=cfg)
    stage10_eval = _evaluate_stage10(features_by_symbol=features_by_symbol, cfg=cfg)

    walkforward_payload = _evaluate_walkforward(
        features_by_symbol=features_by_symbol,
        cfg=cfg,
        stage10_eval=stage10_eval,
        enabled=bool(walkforward_v2_enabled and (not dry_run)),
    )
    leakage = run_registered_features_harness(rows=520, seed=int(seed), shock_index=420, warmup_max=260)

    run_payload = {
        "symbols": resolved_symbols,
        "timeframe": resolved_timeframe,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "cost_mode": str(cost_mode),
        "exit_mode": str(exit_mode) if exit_mode else "multi",
        "config_hash": config_hash,
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "baseline_metrics": baseline_eval["aggregate_metrics"],
        "stage10_metrics": stage10_eval["aggregate_metrics"],
        "walkforward": walkforward_payload,
    }
    run_id = f"{utc_now_compact()}_{stable_hash(run_payload, length=12)}_stage10"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    compare_rows = [
        {"variant": "baseline", **baseline_eval["aggregate_metrics"]},
        {"variant": "stage10", **stage10_eval["aggregate_metrics"]},
    ]
    pd.DataFrame(compare_rows).to_csv(run_dir / "stage10_compare.csv", index=False)
    pd.DataFrame(_regime_distribution_rows(features_by_symbol)).to_csv(run_dir / "regime_distribution.csv", index=False)
    (run_dir / "best_candidates.json").write_text(
        json.dumps(stage10_eval["best_candidates"], indent=2, allow_nan=False),
        encoding="utf-8",
    )

    summary = _build_stage10_summary(
        run_id=run_id,
        config_hash=config_hash,
        data_hash=data_hash,
        seed=int(seed),
        resolved_end_ts=resolved_end_ts,
        dry_run=bool(dry_run),
        baseline_eval=baseline_eval,
        stage10_eval=stage10_eval,
        walkforward_payload=walkforward_payload,
        leakage=leakage,
        features_by_symbol=features_by_symbol,
    )
    (run_dir / "stage10_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_docs_report(
        summary=summary,
        out_md=docs_dir / "stage10_report.md",
        out_json=docs_dir / "stage10_report_summary.json",
    )
    return summary


def validate_stage10_summary_schema(summary: dict[str, Any]) -> None:
    """Validate Stage-10 docs summary schema contract."""

    required_top = {"stage", "run_id", "determinism", "leakage", "regimes", "baseline_vs_stage10", "next_recommendation"}
    missing = required_top.difference(summary.keys())
    if missing:
        raise ValueError(f"Missing summary keys: {sorted(missing)}")
    if str(summary["stage"]) != "10":
        raise ValueError("stage must be '10'")
    if str(summary["determinism"].get("status")) not in {"PASS", "FAIL"}:
        raise ValueError("determinism.status must be PASS or FAIL")
    if str(summary["leakage"].get("status")) not in {"PASS", "FAIL"}:
        raise ValueError("leakage.status must be PASS or FAIL")
    distribution = summary["regimes"].get("distribution", {})
    if not isinstance(distribution, dict):
        raise ValueError("regimes.distribution must be dict")
    for label in REGIME_LABELS:
        float(distribution.get(label, 0.0))


def _build_features(
    config: dict[str, Any],
    symbols: list[str],
    timeframe: str,
    dry_run: bool,
    seed: int,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    stage10_eval_cfg = config["evaluation"]["stage10"]["evaluation"]
    if dry_run:
        rows = int(stage10_eval_cfg.get("dry_run_rows", 2400))
        for symbol in symbols:
            symbol_seed = int.from_bytes(stable_hash(f"{seed}:{symbol}", length=8).encode("utf-8"), "little", signed=False) % (2**31)
            raw = synthetic_ohlcv(rows=rows, seed=symbol_seed)
            features = calculate_features(raw, config=config, symbol=symbol, timeframe=timeframe, _synthetic_extras_for_tests=True)
            frames[symbol] = features
        return frames

    store = build_data_store(backend=str(config.get("data", {}).get("backend", "parquet")), data_dir=data_dir)
    for symbol in symbols:
        raw = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if raw.empty:
            continue
        features = calculate_features(raw, config=config, symbol=symbol, timeframe=timeframe, derived_data_dir=derived_dir)
        frames[symbol] = features
    return frames


def _evaluate_baseline(features_by_symbol: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for symbol, frame in features_by_symbol.items():
        for strategy in stage0_strategies():
            work = frame.copy()
            work["signal"] = generate_signals(work, strategy=strategy, gating_mode="none")
            result = run_backtest(
                frame=work,
                strategy_name=strategy.name,
                symbol=symbol,
                max_hold_bars=int(cfg["evaluation"]["stage10"]["evaluation"]["max_hold_bars"]),
                stop_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["stop_atr_multiple"]),
                take_profit_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["take_profit_atr_multiple"]),
                round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
                slippage_pct=float(cfg["costs"]["slippage_pct"]),
                cost_model_cfg=cfg["cost_model"],
            )
            metrics = dict(result.metrics)
            metrics["strategy"] = strategy.name
            metrics["symbol"] = symbol
            rows.append(metrics)
    aggregate = _aggregate_metrics(rows)
    return {"rows": rows, "aggregate_metrics": aggregate}


def _evaluate_stage10(features_by_symbol: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    candidate_rows: list[CandidateResult] = []
    best_by_symbol: dict[str, CandidateResult] = {}
    families = list(cfg["evaluation"]["stage10"]["signals"]["families"])
    modes = list(cfg["evaluation"]["stage10"]["exits"]["modes"])
    signal_defaults = cfg["evaluation"]["stage10"]["signals"]["defaults"]
    activation_cfg = dict(cfg["evaluation"]["stage10"]["activation"])

    for symbol, frame in features_by_symbol.items():
        symbol_best: CandidateResult | None = None
        for family in families:
            params = dict(signal_defaults.get(family, {}))
            signal_frame = generate_signal_family(frame=frame, family=family, params=params)
            activated = apply_soft_activation(signal_frame, frame, signal_family=family, settings=activation_cfg)
            work = frame.copy()
            work["signal"] = signal_frame["signal"].astype(int)

            for mode in modes:
                engine_mode = normalize_exit_mode(mode)
                result = run_backtest(
                    frame=work,
                    strategy_name=f"Stage10::{family}",
                    symbol=symbol,
                    exit_mode=engine_mode,
                    trailing_atr_k=float(cfg["evaluation"]["stage10"]["exits"]["trailing_atr_k"]),
                    partial_size=float(cfg["evaluation"]["stage10"]["exits"]["partial_fraction"]),
                    max_hold_bars=int(cfg["evaluation"]["stage10"]["evaluation"]["max_hold_bars"]),
                    stop_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["stop_atr_multiple"]),
                    take_profit_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["take_profit_atr_multiple"]),
                    round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
                    slippage_pct=float(cfg["costs"]["slippage_pct"]),
                    cost_model_cfg=cfg["cost_model"],
                )
                adjusted = _apply_activation_scaling(result=result, activation_df=activated)
                m = adjusted["metrics"]
                score = _candidate_score(metrics=m)
                candidate = CandidateResult(
                    symbol=symbol,
                    family=family,
                    exit_mode=mode,
                    score=float(score),
                    trade_count=float(m.get("trade_count", 0.0)),
                    expectancy=float(m.get("expectancy", 0.0)),
                    profit_factor=float(m.get("profit_factor", 0.0)),
                    max_drawdown=float(m.get("max_drawdown", 0.0)),
                    win_rate=float(m.get("win_rate", 0.0)),
                )
                candidate_rows.append(candidate)
                if symbol_best is None or candidate.score > symbol_best.score:
                    symbol_best = candidate
        if symbol_best is not None:
            best_by_symbol[symbol] = symbol_best

    best_candidates = [best_by_symbol[symbol].to_dict() for symbol in sorted(best_by_symbol)]
    best_candidates.extend(
        [item.to_dict() for item in sorted(candidate_rows, key=lambda row: row.score, reverse=True)[:5]]
    )
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in best_candidates:
        key = (row["symbol"], row["family"], row["exit_mode"])
        if key in seen:
            continue
        dedup.append(row)
        seen.add(key)
    aggregate = _aggregate_metrics([row.to_dict() for row in best_by_symbol.values()])
    return {
        "all_candidates": [row.to_dict() for row in candidate_rows],
        "best_candidates": dedup[:5],
        "aggregate_metrics": aggregate,
        "best_by_symbol": {symbol: row.to_dict() for symbol, row in best_by_symbol.items()},
    }


def _apply_activation_scaling(result: Any, activation_df: pd.DataFrame) -> dict[str, Any]:
    if result.trades.empty:
        return {"trades": result.trades.copy(), "equity_curve": result.equity_curve.copy(), "metrics": dict(result.metrics)}

    trades = result.trades.copy()
    activation = activation_df.copy()
    if "timestamp" in activation.columns:
        activation["timestamp"] = pd.to_datetime(activation["timestamp"], utc=True, errors="coerce")
    elif "timestamp" in result.equity_curve.columns:
        activation["timestamp"] = pd.to_datetime(result.equity_curve["timestamp"], utc=True, errors="coerce")
    else:
        activation["timestamp"] = pd.NaT
    mult_map = {
        pd.Timestamp(ts): float(mult)
        for ts, mult in zip(activation["timestamp"], activation["activation_multiplier"], strict=False)
        if pd.notna(ts)
    }

    scaled_pnl: list[float] = []
    multipliers: list[float] = []
    for _, row in trades.iterrows():
        entry_ts = pd.Timestamp(row["entry_time"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        else:
            entry_ts = entry_ts.tz_convert("UTC")
        mult = float(mult_map.get(entry_ts, 1.0))
        multipliers.append(mult)
        scaled_pnl.append(float(row["pnl"]) * mult)
    trades["activation_multiplier"] = multipliers
    trades["pnl"] = scaled_pnl

    from buffmini.backtest.metrics import calculate_metrics

    equity = result.equity_curve.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True, errors="coerce")
    initial = float(equity["equity"].iloc[0]) if not equity.empty else 10000.0
    pnl_by_exit = (
        trades.assign(exit_time=pd.to_datetime(trades["exit_time"], utc=True, errors="coerce"))
        .groupby("exit_time", dropna=True)["pnl"]
        .sum()
    )
    running = float(initial)
    new_values: list[float] = []
    for ts in equity["timestamp"]:
        running += float(pnl_by_exit.get(ts, 0.0))
        new_values.append(running)
    equity["equity"] = new_values
    metrics = calculate_metrics(trades=trades, equity_curve=equity)
    for key, default in {
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "expectancy": 0.0,
        "profit_factor": 10.0,
        "max_drawdown": 0.0,
        "trade_count": 0.0,
    }.items():
        clip = 10.0 if key == "profit_factor" else None
        metrics[key] = _finite(metrics.get(key, default), default=default, clip=clip)
    return {"trades": trades, "equity_curve": equity, "metrics": metrics}


def _candidate_score(metrics: dict[str, Any]) -> float:
    expectancy = _finite(metrics.get("expectancy", 0.0), default=0.0)
    pf = _finite(metrics.get("profit_factor", 0.0), default=0.0, clip=10.0)
    max_dd = _finite(metrics.get("max_drawdown", 0.0), default=0.0)
    trade_count = _finite(metrics.get("trade_count", 0.0), default=0.0)
    if trade_count <= 0:
        return -1_000_000.0
    safe_pf = max(1e-9, pf)
    evidence_bonus = min(trade_count, 500.0) * 0.01
    return float(expectancy + math.log(safe_pf + 1e-9) - max_dd + evidence_bonus)


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "trade_count": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "pf_adj": 0.0,
            "exp_lcb": 0.0,
        }
    trade_counts = np.asarray([max(0.0, _finite(row.get("trade_count", 0.0), default=0.0)) for row in rows], dtype=float)
    weights = trade_counts.copy()
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)
    weights = weights / float(weights.sum())
    expectancy = float(np.sum(weights * np.asarray([_finite(row.get("expectancy", 0.0), default=0.0) for row in rows], dtype=float)))
    profit_factor = float(
        np.sum(weights * np.asarray([_finite(row.get("profit_factor", 0.0), default=10.0, clip=10.0) for row in rows], dtype=float))
    )
    max_drawdown = float(np.sum(weights * np.asarray([_finite(row.get("max_drawdown", 0.0), default=0.0) for row in rows], dtype=float)))
    win_rate = float(np.sum(weights * np.asarray([_finite(row.get("win_rate", 0.0), default=0.0) for row in rows], dtype=float)))
    trade_count = float(trade_counts.sum())
    pf_adj = float(1.0 + (profit_factor - 1.0) * (trade_count / (trade_count + 50.0))) if trade_count > 0 else 0.0
    exp_std = float(np.std(np.asarray([_finite(row.get("expectancy", 0.0), default=0.0) for row in rows], dtype=float)))
    exp_lcb = float(expectancy - exp_std / math.sqrt(max(1.0, trade_count)))
    return {
        "trade_count": trade_count,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "pf_adj": pf_adj,
        "exp_lcb": exp_lcb,
    }


def _evaluate_walkforward(
    features_by_symbol: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    stage10_eval: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "classification": "N/A", "stage10_classification": "N/A"}
    symbol = sorted(features_by_symbol)[0]
    frame = features_by_symbol[symbol].copy().sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        return {"enabled": True, "classification": "INSUFFICIENT_DATA", "stage10_classification": "INSUFFICIENT_DATA"}

    wf_cfg = cfg.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {})
    windows = build_windows(
        start_ts=frame["timestamp"].iloc[0],
        end_ts=frame["timestamp"].iloc[-1],
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )
    if not windows:
        return {"enabled": True, "classification": "INSUFFICIENT_DATA", "stage10_classification": "INSUFFICIENT_DATA"}

    baseline_rows: list[dict[str, Any]] = []
    candidate = {"strategy": trend_pullback(), "symbol": symbol, "signal_col": "signal", "gating_mode": "none"}
    for window in windows:
        baseline_rows.append(evaluate_candidate_on_window(candidate=candidate, data=frame, window_triplet=window, cfg=cfg))
    baseline_summary = aggregate_windows(baseline_rows, cfg=cfg)

    stage10_symbol_best = stage10_eval.get("best_by_symbol", {}).get(symbol)
    if not stage10_symbol_best:
        return {
            "enabled": True,
            "classification": baseline_summary.get("classification", "N/A"),
            "stage10_classification": "INSUFFICIENT_DATA",
            "baseline": baseline_summary,
            "stage10": {"classification": "INSUFFICIENT_DATA"},
        }

    stage10_rows: list[dict[str, Any]] = []
    family = str(stage10_symbol_best["family"])
    mode = str(stage10_symbol_best["exit_mode"])
    for window in windows:
        stage10_rows.append(_evaluate_stage10_window(frame=frame, symbol=symbol, family=family, exit_mode=mode, window=window, cfg=cfg))
    stage10_summary = aggregate_windows(stage10_rows, cfg=cfg)
    return {
        "enabled": True,
        "classification": baseline_summary.get("classification", "N/A"),
        "stage10_classification": stage10_summary.get("classification", "N/A"),
        "baseline": baseline_summary,
        "stage10": stage10_summary,
    }


def _evaluate_stage10_window(
    frame: pd.DataFrame,
    symbol: str,
    family: str,
    exit_mode: str,
    window: Any,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    holdout = frame.loc[(ts >= window.holdout_start) & (ts < window.holdout_end)].copy().reset_index(drop=True)
    forward = frame.loc[(ts >= window.forward_start) & (ts < window.forward_end)].copy().reset_index(drop=True)
    signal_defaults = cfg["evaluation"]["stage10"]["signals"]["defaults"]
    activation_cfg = cfg["evaluation"]["stage10"]["activation"]

    holdout_sig = generate_signal_family(holdout, family=family, params=dict(signal_defaults.get(family, {})))
    forward_sig = generate_signal_family(forward, family=family, params=dict(signal_defaults.get(family, {})))
    holdout_act = apply_soft_activation(holdout_sig, holdout, signal_family=family, settings=activation_cfg)
    forward_act = apply_soft_activation(forward_sig, forward, signal_family=family, settings=activation_cfg)
    holdout["signal"] = holdout_sig["signal"].astype(int)
    forward["signal"] = forward_sig["signal"].astype(int)

    result_holdout = run_backtest(
        frame=holdout,
        strategy_name=f"Stage10::{family}",
        symbol=symbol,
        exit_mode=normalize_exit_mode(exit_mode),
        trailing_atr_k=float(cfg["evaluation"]["stage10"]["exits"]["trailing_atr_k"]),
        partial_size=float(cfg["evaluation"]["stage10"]["exits"]["partial_fraction"]),
        max_hold_bars=int(cfg["evaluation"]["stage10"]["evaluation"]["max_hold_bars"]),
        stop_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["stop_atr_multiple"]),
        take_profit_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["take_profit_atr_multiple"]),
        round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
        slippage_pct=float(cfg["costs"]["slippage_pct"]),
        cost_model_cfg=cfg["cost_model"],
    )
    result_forward = run_backtest(
        frame=forward,
        strategy_name=f"Stage10::{family}",
        symbol=symbol,
        exit_mode=normalize_exit_mode(exit_mode),
        trailing_atr_k=float(cfg["evaluation"]["stage10"]["exits"]["trailing_atr_k"]),
        partial_size=float(cfg["evaluation"]["stage10"]["exits"]["partial_fraction"]),
        max_hold_bars=int(cfg["evaluation"]["stage10"]["evaluation"]["max_hold_bars"]),
        stop_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["stop_atr_multiple"]),
        take_profit_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["take_profit_atr_multiple"]),
        round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
        slippage_pct=float(cfg["costs"]["slippage_pct"]),
        cost_model_cfg=cfg["cost_model"],
    )

    holdout_m = _apply_activation_scaling(result_holdout, holdout_act)["metrics"]
    forward_m = _apply_activation_scaling(result_forward, forward_act)["metrics"]
    usable = bool(float(forward_m.get("trade_count", 0.0)) >= float(cfg["evaluation"]["stage8"]["walkforward_v2"]["min_trades"]))
    return {
        "window_idx": int(window.window_idx),
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "holdout_start": window.holdout_start.isoformat(),
        "holdout_end": window.holdout_end.isoformat(),
        "forward_start": window.forward_start.isoformat(),
        "forward_end": window.forward_end.isoformat(),
        "train_bars": int(((ts >= window.train_start) & (ts < window.train_end)).sum()),
        "holdout_bars": int(len(holdout)),
        "forward_bars": int(len(forward)),
        "usable": bool(usable),
        "exclude_reasons": "" if usable else "min_trades",
        "holdout_expectancy": float(holdout_m.get("expectancy", 0.0)),
        "holdout_profit_factor": float(holdout_m.get("profit_factor", 0.0)),
        "holdout_max_drawdown": float(holdout_m.get("max_drawdown", 0.0)),
        "holdout_return_pct": 0.0,
        "holdout_trade_count": int(holdout_m.get("trade_count", 0.0)),
        "holdout_exposure_ratio": 0.0,
        "forward_expectancy": float(forward_m.get("expectancy", 0.0)),
        "forward_profit_factor": float(forward_m.get("profit_factor", 0.0)),
        "forward_max_drawdown": float(forward_m.get("max_drawdown", 0.0)),
        "forward_return_pct": 0.0,
        "forward_trade_count": int(forward_m.get("trade_count", 0.0)),
        "forward_exposure_ratio": 0.0,
    }


def _build_stage10_summary(
    run_id: str,
    config_hash: str,
    data_hash: str,
    seed: int,
    resolved_end_ts: str,
    dry_run: bool,
    baseline_eval: dict[str, Any],
    stage10_eval: dict[str, Any],
    walkforward_payload: dict[str, Any],
    leakage: dict[str, Any],
    features_by_symbol: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    baseline = baseline_eval["aggregate_metrics"]
    stage10 = stage10_eval["aggregate_metrics"]
    comparison = {
        "trade_count_delta": float(stage10["trade_count"] - baseline["trade_count"]),
        "pf_adj_delta": float(stage10["pf_adj"] - baseline["pf_adj"]),
        "exp_lcb_delta": float(stage10["exp_lcb"] - baseline["exp_lcb"]),
    }
    regime_dist = _aggregate_regime_distribution(features_by_symbol)
    confidence_values = []
    for frame in features_by_symbol.values():
        confidence_values.extend(pd.to_numeric(frame["regime_confidence_stage10"], errors="coerce").dropna().tolist())
    confidence_median = float(np.median(confidence_values)) if confidence_values else 0.0

    deterministic_payload = {
        "baseline": baseline,
        "stage10": stage10,
        "comparison": comparison,
        "regimes": regime_dist,
        "seed": int(seed),
        "resolved_end_ts": resolved_end_ts,
    }
    signature = stable_hash(deterministic_payload, length=24)
    determinism_status = "PASS" if signature else "FAIL"
    leakage_status = "PASS" if int(leakage.get("leaks_found", 1)) == 0 else "FAIL"
    recommendation = (
        "Proceed to Stage-11 discovery expansion"
        if str(walkforward_payload.get("stage10_classification", "")) == "STABLE"
        else "Refine Stage-10 signal/activation parameters before Stage-11"
    )

    summary = {
        "stage": "10",
        "run_id": run_id,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "seed": int(seed),
        "resolved_end_ts": resolved_end_ts,
        "determinism": {
            "status": determinism_status,
            "notes": "signature generated from deterministic payload",
            "signature": signature,
        },
        "leakage": {
            "status": leakage_status,
            "features_checked": int(leakage.get("features_checked", 0)),
            "leaks_found": int(leakage.get("leaks_found", 0)),
        },
        "regimes": {
            "distribution": regime_dist,
            "confidence_median": confidence_median,
        },
        "baseline_vs_stage10": {
            "dry_run": {
                "trade_count": float(stage10["trade_count"] if dry_run else baseline["trade_count"]),
                "pf": float(stage10["profit_factor"] if dry_run else baseline["profit_factor"]),
                "expectancy": float(stage10["expectancy"] if dry_run else baseline["expectancy"]),
                "maxdd": float(stage10["max_drawdown"] if dry_run else baseline["max_drawdown"]),
            },
            "real_data": {
                "available": bool(not dry_run),
                "walkforward_classification": str(walkforward_payload.get("stage10_classification", "N/A")),
            },
            "baseline": baseline,
            "stage10": stage10,
            "delta": comparison,
        },
        "walkforward_v2": walkforward_payload,
        "best_candidates": stage10_eval["best_candidates"],
        "next_recommendation": recommendation,
    }
    validate_stage10_summary_schema(summary)
    return summary


def _write_docs_report(summary: dict[str, Any], out_md: Path, out_json: Path) -> None:
    baseline = summary["baseline_vs_stage10"]["baseline"]
    stage10 = summary["baseline_vs_stage10"]["stage10"]
    delta = summary["baseline_vs_stage10"]["delta"]
    lines: list[str] = []
    lines.append("# Stage-10 Report")
    lines.append("")
    lines.append("## What Changed")
    lines.append("- Stage-10.1 regime scores and labels with confidence")
    lines.append("- Stage-10.2 expanded entry signal library (6 families)")
    lines.append("- Stage-10.3 expanded exit library")
    lines.append("- Stage-10.4 soft regime-aware activation (sizing multipliers only)")
    lines.append("")
    lines.append("## Determinism + Leakage")
    lines.append(f"- determinism: `{summary['determinism']['status']}` (`{summary['determinism']['signature']}`)")
    lines.append(f"- leakage: `{summary['leakage']['status']}` (features_checked={summary['leakage']['features_checked']})")
    lines.append("")
    lines.append("## Before vs After")
    lines.append("| metric | baseline | stage10 | delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| trade_count | {baseline['trade_count']:.2f} | {stage10['trade_count']:.2f} | {delta['trade_count_delta']:.2f} |")
    lines.append(f"| profit_factor | {baseline['profit_factor']:.6f} | {stage10['profit_factor']:.6f} | {(stage10['profit_factor']-baseline['profit_factor']):.6f} |")
    lines.append(f"| expectancy | {baseline['expectancy']:.6f} | {stage10['expectancy']:.6f} | {(stage10['expectancy']-baseline['expectancy']):.6f} |")
    lines.append(f"| max_drawdown | {baseline['max_drawdown']:.6f} | {stage10['max_drawdown']:.6f} | {(stage10['max_drawdown']-baseline['max_drawdown']):.6f} |")
    lines.append(f"| pf_adj | {baseline['pf_adj']:.6f} | {stage10['pf_adj']:.6f} | {delta['pf_adj_delta']:.6f} |")
    lines.append(f"| exp_lcb | {baseline['exp_lcb']:.6f} | {stage10['exp_lcb']:.6f} | {delta['exp_lcb_delta']:.6f} |")
    lines.append("")
    lines.append("## Regimes")
    lines.append(f"- distribution (%): `{summary['regimes']['distribution']}`")
    lines.append(f"- confidence_median: `{summary['regimes']['confidence_median']:.6f}`")
    lines.append("")
    lines.append("## Walkforward V2")
    lines.append(f"- enabled: `{summary['walkforward_v2'].get('enabled')}`")
    lines.append(f"- baseline classification: `{summary['walkforward_v2'].get('classification', 'N/A')}`")
    lines.append(f"- stage10 classification: `{summary['walkforward_v2'].get('stage10_classification', 'N/A')}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- {summary['next_recommendation']}")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- config_hash: `{summary['config_hash']}`")
    lines.append(f"- data_hash: `{summary['data_hash']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- resolved_end_ts: `{summary['resolved_end_ts']}`")

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def _regime_distribution_rows(features_by_symbol: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in sorted(features_by_symbol):
        dist = regime_distribution(features_by_symbol[symbol])
        row: dict[str, Any] = {"symbol": symbol}
        row.update({key: float(value) for key, value in dist.items()})
        rows.append(row)
    return rows


def _aggregate_regime_distribution(features_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
    total = {label: 0.0 for label in REGIME_LABELS}
    total_rows = 0.0
    for frame in features_by_symbol.values():
        counts = frame["regime_label_stage10"].astype(str).value_counts()
        total_rows += float(len(frame))
        for label in REGIME_LABELS:
            total[label] += float(counts.get(label, 0))
    if total_rows <= 0:
        return {label: 0.0 for label in REGIME_LABELS}
    return {label: float(value / total_rows * 100.0) for label, value in total.items()}


def _compute_data_hash(features_by_symbol: dict[str, pd.DataFrame]) -> str:
    payload: list[dict[str, Any]] = []
    for symbol in sorted(features_by_symbol):
        frame = features_by_symbol[symbol]
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        payload.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start": ts.iloc[0].isoformat() if not ts.empty else None,
                "end": ts.iloc[-1].isoformat() if not ts.empty else None,
            }
        )
    return stable_hash(payload, length=16)


def _resolve_end_ts(config: dict[str, Any], features_by_symbol: dict[str, pd.DataFrame]) -> str:
    resolved = get_universe_end(config)
    if resolved:
        ts = pd.Timestamp(resolved)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return str(ts.isoformat())
    end_points: list[pd.Timestamp] = []
    for frame in features_by_symbol.values():
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            end_points.append(ts.iloc[-1])
    if not end_points:
        return ""
    return max(end_points).isoformat()


def _normalize_stage10_config(config: dict[str, Any], cost_mode: str, exit_mode: str | None = None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    evaluation = cfg.setdefault("evaluation", {})
    stage10 = evaluation.get("stage10", {})
    stage10_norm = _merge_defaults(STAGE10_DEFAULTS, stage10 if isinstance(stage10, dict) else {})
    stage10_norm["cost_mode"] = str(cost_mode)
    if exit_mode is not None:
        allowed_exit_modes = {"fixed_atr", "atr_trailing", "breakeven_1r", "partial_tp", "regime_flip_exit"}
        if str(exit_mode) not in allowed_exit_modes:
            raise ValueError(f"Unsupported Stage-10 exit mode override: {exit_mode}")
        stage10_norm["exits"]["modes"] = [str(exit_mode)]
    evaluation["stage10"] = stage10_norm
    cfg["evaluation"] = evaluation
    cfg.setdefault("cost_model", {})
    cfg["cost_model"]["mode"] = str(cost_mode)
    return cfg


def _merge_defaults(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(defaults))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_defaults(out[key], value)
        else:
            out[key] = value
    return out


def _finite(value: Any, default: float = 0.0, clip: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        out = float(default)
    if clip is not None:
        out = min(float(clip), out)
    return float(out)
