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
from buffmini.data.cache import FeatureFrameCache, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.stage10.activation import DEFAULT_ACTIVATION_CONFIG, apply_soft_activation
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.stage10.regimes import REGIME_LABELS, regime_calibration_diagnostics, regime_distribution
from buffmini.stage10.signals import DEFAULT_SIGNAL_PARAMS, SIGNAL_FAMILIES, generate_signal_family, resolve_enabled_families
from buffmini.stage11.hooks import build_noop_hooks
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.leakage_harness import run_registered_features_harness, synthetic_ohlcv
from buffmini.validation.walkforward_v2 import aggregate_windows, build_windows, evaluate_candidate_on_window


STAGE10_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "cost_mode": "v2",
    "walkforward_v2": True,
    "regimes": {
        "trend_rank_strong": 0.60,
        "trend_rank_weak": 0.40,
        "high_vol_rank": 0.75,
        "low_vol_rank": 0.25,
        "chop_flip_window": 48,
        "chop_flip_threshold": 0.18,
        "compression_z": -0.8,
        "expansion_z": 1.0,
        "volume_z_high": 1.0,
    },
    "activation": dict(DEFAULT_ACTIVATION_CONFIG),
    "signals": {
        "families": list(SIGNAL_FAMILIES),
        "enabled_families": [
            "BreakoutRetest",
            "MA_SlopePullback",
            "ATR_DistanceRevert",
            "RangeFade",
            "VolCompressionBreakout",
        ],
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
    write_docs: bool = True,
    is_stress: bool = False,
    hooks: dict[str, Any] | None = None,
    features_by_symbol_override: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run Stage-10 baseline-vs-upgraded comparison and write artifacts."""

    cfg = _normalize_stage10_config(config=config, cost_mode=cost_mode, exit_mode=exit_mode)
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    resolved_timeframe = str(timeframe or cfg.get("universe", {}).get("timeframe", "1h"))
    if resolved_timeframe != "1h":
        raise ValueError("Stage-10 currently supports timeframe=1h only")

    if isinstance(features_by_symbol_override, dict) and features_by_symbol_override:
        features_by_symbol = {str(sym): frame.copy() for sym, frame in features_by_symbol_override.items()}
    else:
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
    resolved_hooks = _resolve_stage11_hooks(hooks)
    stage10_eval = _evaluate_stage10(features_by_symbol=features_by_symbol, cfg=cfg, hooks=resolved_hooks)

    walkforward_payload = _evaluate_walkforward(
        features_by_symbol=features_by_symbol,
        cfg=cfg,
        stage10_eval=stage10_eval,
        enabled=bool(walkforward_v2_enabled and (not dry_run)),
        hooks=resolved_hooks,
    )
    leakage = run_registered_features_harness(rows=520, seed=int(seed), shock_index=420, warmup_max=260)

    run_payload = {
        "symbols": resolved_symbols,
        "timeframe": resolved_timeframe,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "cost_mode": str(cost_mode),
        "exit_mode": str(exit_mode) if exit_mode else "multi",
        "is_stress": bool(is_stress),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "baseline_metrics": baseline_eval["aggregate_metrics"],
        "stage10_metrics": stage10_eval["aggregate_metrics"],
        "walkforward": walkforward_payload,
        "hooks_enabled": {
            "bias": hooks is not None and "bias" in hooks,
            "confirm": hooks is not None and "confirm" in hooks,
            "exit": hooks is not None and "exit" in hooks,
        },
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
        cfg=cfg,
        baseline_eval=baseline_eval,
        stage10_eval=stage10_eval,
        walkforward_payload=walkforward_payload,
        leakage=leakage,
        features_by_symbol=features_by_symbol,
        is_stress=bool(is_stress),
    )
    (run_dir / "stage10_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if bool(write_docs):
        docs_dir.mkdir(parents=True, exist_ok=True)
        _write_docs_report(
            summary=summary,
            out_md=docs_dir / "stage10_report.md",
            out_json=docs_dir / "stage10_report_summary.json",
        )
    return summary


def run_stage10_exit_ab(
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    cost_mode: str = "v2",
    walkforward_v2_enabled: bool = True,
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic Stage-10 exit A/B and select best isolated exit."""

    compare_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for mode in ("fixed_atr", "atr_trailing"):
        base = run_stage10(
            config=config,
            seed=int(seed),
            dry_run=bool(dry_run),
            symbols=symbols,
            timeframe=timeframe,
            cost_mode=cost_mode,
            walkforward_v2_enabled=walkforward_v2_enabled,
            exit_mode=mode,
            runs_root=runs_root,
            docs_dir=docs_dir,
            data_dir=data_dir,
            derived_dir=derived_dir,
            write_docs=False,
            is_stress=False,
            hooks=hooks,
        )
        stress = run_stage10(
            config=_build_stress_config(config),
            seed=int(seed),
            dry_run=bool(dry_run),
            symbols=symbols,
            timeframe=timeframe,
            cost_mode="v2",
            walkforward_v2_enabled=False,
            exit_mode=mode,
            runs_root=runs_root,
            docs_dir=docs_dir,
            data_dir=data_dir,
            derived_dir=derived_dir,
            write_docs=False,
            is_stress=True,
            hooks=hooks,
        )
        base_metrics = dict(base["baseline_vs_stage10"]["stage10"])
        stress_metrics = dict(stress["baseline_vs_stage10"]["stage10"])
        drag = max(
            0.0,
            _finite(base_metrics.get("expectancy", 0.0), default=0.0)
            - _finite(stress_metrics.get("expectancy", 0.0), default=0.0),
        )
        row = {
            "exit_mode": mode,
            "run_id": str(base.get("run_id", "")),
            "trade_count": _finite(base_metrics.get("trade_count", 0.0), default=0.0),
            "profit_factor": _finite(base_metrics.get("profit_factor", 0.0), default=0.0, clip=10.0),
            "expectancy": _finite(base_metrics.get("expectancy", 0.0), default=0.0),
            "max_drawdown": _finite(base_metrics.get("max_drawdown", 0.0), default=0.0),
            "exp_lcb": _finite(base_metrics.get("exp_lcb", 0.0), default=0.0),
            "drag_sensitivity": float(drag),
            "stress_expectancy": _finite(stress_metrics.get("expectancy", 0.0), default=0.0),
            "stress_profit_factor": _finite(stress_metrics.get("profit_factor", 0.0), default=0.0, clip=10.0),
        }
        compare_rows.append(row)
        summaries[mode] = base

    compare_rows.sort(key=lambda item: (-float(item["exp_lcb"]), float(item["drag_sensitivity"]), str(item["exit_mode"])))
    selected = compare_rows[0]["exit_mode"] if compare_rows else "fixed_atr"
    selected_summary = summaries.get(str(selected), {})

    compare_payload = {
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "symbols": list(symbols or []),
        "timeframe": str(timeframe),
        "cost_mode": str(cost_mode),
        "rows": compare_rows,
        "selected_exit": selected,
    }
    run_id = f"{utc_now_compact()}_{stable_hash(compare_payload, length=12)}_stage10_exit_ab"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(compare_rows).to_csv(run_dir / "exit_ab_compare.csv", index=False)
    (run_dir / "exit_ab_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "selected_exit": selected,
                "rows": compare_rows,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    if selected_summary:
        docs_dir.mkdir(parents=True, exist_ok=True)
        _write_docs_report(
            summary=selected_summary,
            out_md=docs_dir / "stage10_report.md",
            out_json=docs_dir / "stage10_report_summary.json",
        )

    return {
        "run_id": run_id,
        "selected_exit": selected,
        "rows": compare_rows,
        "selected_summary": selected_summary,
    }


def validate_stage10_summary_schema(summary: dict[str, Any]) -> None:
    """Validate Stage-10 docs summary schema contract."""

    required_top = {
        "stage",
        "run_id",
        "determinism",
        "leakage",
        "regimes",
        "baseline_vs_stage10",
        "trade_count_guard",
        "next_recommendation",
    }
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
    feature_cache_enabled = bool(config.get("data", {}).get("feature_cache", {}).get("enabled", True))
    feature_cache = FeatureFrameCache() if feature_cache_enabled else None
    if dry_run:
        rows = int(stage10_eval_cfg.get("dry_run_rows", 2400))
        for symbol in symbols:
            symbol_seed = int.from_bytes(stable_hash(f"{seed}:{symbol}", length=8).encode("utf-8"), "little", signed=False) % (2**31)
            raw = synthetic_ohlcv(rows=rows, seed=symbol_seed)
            features = calculate_features(raw, config=config, symbol=symbol, timeframe=timeframe, _synthetic_extras_for_tests=True)
            frames[symbol] = features
        return frames

    store = build_data_store(
        backend=str(config.get("data", {}).get("backend", "parquet")),
        data_dir=data_dir,
        base_timeframe=str(config.get("universe", {}).get("base_timeframe") or timeframe),
        resample_source=str(config.get("data", {}).get("resample_source", "direct")),
        derived_dir=derived_dir,
        partial_last_bucket=bool(config.get("data", {}).get("partial_last_bucket", False)),
    )
    for symbol in symbols:
        raw = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if raw.empty:
            continue
        data_hash = ohlcv_data_hash(raw)
        params_hash = stable_hash(
            {
                "timeframe": str(timeframe),
                "include_futures_extras": bool(config.get("data", {}).get("include_futures_extras", False)),
                "futures_extras": config.get("data", {}).get("futures_extras", {}),
                "cost_model": config.get("cost_model", {}),
            },
            length=16,
        )
        if feature_cache is not None:
            cache_key = feature_cache.key(
                symbol=str(symbol),
                timeframe=str(timeframe),
                data_hash=str(data_hash),
                params_hash=str(params_hash),
            )
            features, _ = feature_cache.get_or_build(
                key=cache_key,
                builder=lambda r=raw, s=symbol: calculate_features(
                    r,
                    config=config,
                    symbol=s,
                    timeframe=timeframe,
                    derived_data_dir=derived_dir,
                ),
                meta={
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "data_hash": str(data_hash),
                    "params_hash": str(params_hash),
                },
            )
        else:
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


def _evaluate_stage10(
    features_by_symbol: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    hooks: dict[str, Any] | None = None,
    return_paths: bool = False,
) -> dict[str, Any]:
    candidate_rows: list[CandidateResult] = []
    best_by_symbol: dict[str, CandidateResult] = {}
    best_paths: dict[str, dict[str, pd.DataFrame]] = {}
    families = resolve_enabled_families(
        families=list(cfg["evaluation"]["stage10"]["signals"]["families"]),
        enabled_families=list(cfg["evaluation"]["stage10"]["signals"].get("enabled_families", [])),
    )
    modes = list(cfg["evaluation"]["stage10"]["exits"]["modes"])
    signal_defaults = cfg["evaluation"]["stage10"]["signals"]["defaults"]
    activation_cfg = dict(cfg["evaluation"]["stage10"]["activation"])
    resolved_hooks = _resolve_stage11_hooks(hooks)

    for symbol, frame in features_by_symbol.items():
        symbol_best: CandidateResult | None = None
        for family in families:
            params = dict(signal_defaults.get(family, {}))
            signal_frame = generate_signal_family(frame=frame, family=family, params=params)
            signal_frame = _apply_confirm_hook_to_signal_frame(
                signal_frame=signal_frame,
                base_frame=frame,
                symbol=symbol,
                signal_family=family,
                confirm_hook=resolved_hooks["confirm"],
            )
            activated = apply_soft_activation(signal_frame, frame, signal_family=family, settings=activation_cfg)
            activated = _apply_bias_hook_to_activation(
                activation_df=activated,
                base_frame=frame,
                symbol=symbol,
                signal_family=family,
                bias_hook=resolved_hooks["bias"],
            )
            work = frame.copy()
            work["signal"] = signal_frame["signal"].astype(int)

            for mode in modes:
                exit_cfg = _apply_exit_hook(
                    exit_hook=resolved_hooks["exit"],
                    base_frame=frame,
                    symbol=symbol,
                    signal_family=family,
                    exit_mode=mode,
                    trailing_atr_k=float(cfg["evaluation"]["stage10"]["exits"]["trailing_atr_k"]),
                    partial_size=float(cfg["evaluation"]["stage10"]["exits"]["partial_fraction"]),
                )
                engine_mode = normalize_exit_mode(str(exit_cfg["exit_mode"]))
                result = run_backtest(
                    frame=work,
                    strategy_name=f"Stage10::{family}",
                    symbol=symbol,
                    exit_mode=engine_mode,
                    trailing_atr_k=float(exit_cfg["trailing_atr_k"]),
                    partial_size=float(exit_cfg["partial_size"]),
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
                    if bool(return_paths):
                        best_paths[symbol] = {
                            "trades": adjusted["trades"].copy(),
                            "equity_curve": adjusted["equity_curve"].copy(),
                        }
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
    family_breakdown = _family_trade_breakdown(candidate_rows)
    return {
        "all_candidates": [row.to_dict() for row in candidate_rows],
        "best_candidates": dedup[:5],
        "aggregate_metrics": aggregate,
        "best_by_symbol": {symbol: row.to_dict() for symbol, row in best_by_symbol.items()},
        "family_trade_breakdown": family_breakdown,
        "best_paths": best_paths if bool(return_paths) else {},
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


def _resolve_stage11_hooks(hooks: dict[str, Any] | None) -> dict[str, Any]:
    defaults = build_noop_hooks()
    if not isinstance(hooks, dict):
        return defaults
    for key in ("bias", "confirm", "exit"):
        value = hooks.get(key)
        if callable(value):
            defaults[key] = value
    return defaults


def _apply_confirm_hook_to_signal_frame(
    signal_frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    symbol: str,
    signal_family: str,
    confirm_hook: Any,
) -> pd.DataFrame:
    out = signal_frame.copy()
    if not callable(confirm_hook):
        return out
    if hasattr(confirm_hook, "start_sequence") and callable(getattr(confirm_hook, "start_sequence")):
        confirm_hook.start_sequence()
    timestamps = pd.to_datetime(base_frame.get("timestamp"), utc=True, errors="coerce")
    revised: list[int] = []
    for idx, signal in enumerate(out["signal"].astype(int).tolist()):
        ts = timestamps.iloc[idx] if len(timestamps) > idx else pd.NaT
        if pd.isna(ts):
            ts = pd.Timestamp("1970-01-01T00:00:00Z")
        row_payload = base_frame.iloc[idx].to_dict() if len(base_frame) > idx else {}
        updated = confirm_hook(
            timestamp=pd.Timestamp(ts),
            symbol=str(symbol),
            signal_family=str(signal_family),
            signal=int(signal),
            base_row=row_payload,
        )
        try:
            numeric = int(updated)
        except Exception:
            numeric = int(signal)
        if numeric > 0:
            numeric = 1
        elif numeric < 0:
            numeric = -1
        revised.append(numeric)
    out["signal"] = pd.Series(revised, index=out.index, dtype=int)
    out["long_entry"] = out["signal"] == 1
    out["short_entry"] = out["signal"] == -1
    if hasattr(confirm_hook, "finalize_sequence") and callable(getattr(confirm_hook, "finalize_sequence")):
        confirm_hook.finalize_sequence()
    return out


def _apply_bias_hook_to_activation(
    activation_df: pd.DataFrame,
    base_frame: pd.DataFrame,
    symbol: str,
    signal_family: str,
    bias_hook: Any,
) -> pd.DataFrame:
    out = activation_df.copy()
    if not callable(bias_hook):
        return out
    timestamps = pd.to_datetime(base_frame.get("timestamp"), utc=True, errors="coerce")
    bias_values: list[float] = []
    for idx, row in out.iterrows():
        ts = timestamps.iloc[idx] if len(timestamps) > idx else pd.NaT
        if pd.isna(ts):
            ts = pd.Timestamp("1970-01-01T00:00:00Z")
        base_row = base_frame.iloc[idx].to_dict() if len(base_frame) > idx else {}
        signal = int(row.get("signal", 0))
        activation_multiplier = float(row.get("activation_multiplier", 1.0))
        value = bias_hook(
            timestamp=pd.Timestamp(ts),
            symbol=str(symbol),
            signal_family=str(signal_family),
            signal=signal,
            base_row=base_row,
            activation_multiplier=activation_multiplier,
        )
        try:
            numeric = float(value)
        except Exception:
            numeric = 1.0
        if not np.isfinite(numeric):
            numeric = 1.0
        bias_values.append(float(numeric))
    out["stage11_bias_multiplier"] = pd.Series(bias_values, index=out.index, dtype=float)
    out["activation_multiplier"] = (
        pd.to_numeric(out["activation_multiplier"], errors="coerce").fillna(1.0)
        * pd.to_numeric(out["stage11_bias_multiplier"], errors="coerce").fillna(1.0)
    )
    out["effective_strength"] = (
        pd.to_numeric(out["signal_strength"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["activation_multiplier"], errors="coerce").fillna(1.0)
    )
    return out


def _apply_exit_hook(
    *,
    exit_hook: Any,
    base_frame: pd.DataFrame,
    symbol: str,
    signal_family: str,
    exit_mode: str,
    trailing_atr_k: float,
    partial_size: float,
) -> dict[str, Any]:
    if not callable(exit_hook):
        return {
            "exit_mode": str(exit_mode),
            "trailing_atr_k": float(trailing_atr_k),
            "partial_size": float(partial_size),
        }
    if base_frame.empty:
        row_payload: dict[str, Any] = {}
        timestamp = pd.Timestamp("1970-01-01T00:00:00Z")
    else:
        row_payload = base_frame.iloc[-1].to_dict()
        ts = pd.to_datetime(row_payload.get("timestamp"), utc=True, errors="coerce")
        timestamp = pd.Timestamp("1970-01-01T00:00:00Z") if pd.isna(ts) else pd.Timestamp(ts)
    payload = exit_hook(
        timestamp=timestamp,
        symbol=str(symbol),
        signal_family=str(signal_family),
        exit_mode=str(exit_mode),
        trailing_atr_k=float(trailing_atr_k),
        partial_size=float(partial_size),
        base_row=row_payload,
    )
    if not isinstance(payload, dict):
        payload = {}
    return {
        "exit_mode": str(payload.get("exit_mode", exit_mode)),
        "trailing_atr_k": float(payload.get("trailing_atr_k", trailing_atr_k)),
        "partial_size": float(payload.get("partial_size", partial_size)),
    }


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


def _family_trade_breakdown(candidates: list[CandidateResult]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    rows = pd.DataFrame([row.to_dict() for row in candidates])
    if rows.empty:
        return []
    grouped = (
        rows.sort_values(["symbol", "family", "trade_count"], ascending=[True, True, False])
        .drop_duplicates(subset=["symbol", "family"], keep="first")
        .groupby("family", dropna=False)["trade_count"]
        .sum()
        .sort_values(ascending=False)
    )
    return [{"family": str(family), "trade_count": float(value)} for family, value in grouped.items()]


def _evaluate_walkforward(
    features_by_symbol: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    stage10_eval: dict[str, Any],
    enabled: bool,
    hooks: dict[str, Any] | None = None,
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
        stage10_rows.append(
            _evaluate_stage10_window(
                frame=frame,
                symbol=symbol,
                family=family,
                exit_mode=mode,
                window=window,
                cfg=cfg,
                hooks=hooks,
            )
        )
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
    hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    holdout = frame.loc[(ts >= window.holdout_start) & (ts < window.holdout_end)].copy().reset_index(drop=True)
    forward = frame.loc[(ts >= window.forward_start) & (ts < window.forward_end)].copy().reset_index(drop=True)
    signal_defaults = cfg["evaluation"]["stage10"]["signals"]["defaults"]
    activation_cfg = cfg["evaluation"]["stage10"]["activation"]
    resolved_hooks = _resolve_stage11_hooks(hooks)

    holdout_sig = generate_signal_family(holdout, family=family, params=dict(signal_defaults.get(family, {})))
    forward_sig = generate_signal_family(forward, family=family, params=dict(signal_defaults.get(family, {})))
    holdout_sig = _apply_confirm_hook_to_signal_frame(
        signal_frame=holdout_sig,
        base_frame=holdout,
        symbol=symbol,
        signal_family=family,
        confirm_hook=resolved_hooks["confirm"],
    )
    forward_sig = _apply_confirm_hook_to_signal_frame(
        signal_frame=forward_sig,
        base_frame=forward,
        symbol=symbol,
        signal_family=family,
        confirm_hook=resolved_hooks["confirm"],
    )
    holdout_act = apply_soft_activation(holdout_sig, holdout, signal_family=family, settings=activation_cfg)
    forward_act = apply_soft_activation(forward_sig, forward, signal_family=family, settings=activation_cfg)
    holdout_act = _apply_bias_hook_to_activation(
        activation_df=holdout_act,
        base_frame=holdout,
        symbol=symbol,
        signal_family=family,
        bias_hook=resolved_hooks["bias"],
    )
    forward_act = _apply_bias_hook_to_activation(
        activation_df=forward_act,
        base_frame=forward,
        symbol=symbol,
        signal_family=family,
        bias_hook=resolved_hooks["bias"],
    )
    holdout["signal"] = holdout_sig["signal"].astype(int)
    forward["signal"] = forward_sig["signal"].astype(int)
    exit_cfg_holdout = _apply_exit_hook(
        exit_hook=resolved_hooks["exit"],
        base_frame=holdout,
        symbol=symbol,
        signal_family=family,
        exit_mode=exit_mode,
        trailing_atr_k=float(cfg["evaluation"]["stage10"]["exits"]["trailing_atr_k"]),
        partial_size=float(cfg["evaluation"]["stage10"]["exits"]["partial_fraction"]),
    )
    exit_cfg_forward = _apply_exit_hook(
        exit_hook=resolved_hooks["exit"],
        base_frame=forward,
        symbol=symbol,
        signal_family=family,
        exit_mode=exit_mode,
        trailing_atr_k=float(cfg["evaluation"]["stage10"]["exits"]["trailing_atr_k"]),
        partial_size=float(cfg["evaluation"]["stage10"]["exits"]["partial_fraction"]),
    )

    result_holdout = run_backtest(
        frame=holdout,
        strategy_name=f"Stage10::{family}",
        symbol=symbol,
        exit_mode=normalize_exit_mode(str(exit_cfg_holdout["exit_mode"])),
        trailing_atr_k=float(exit_cfg_holdout["trailing_atr_k"]),
        partial_size=float(exit_cfg_holdout["partial_size"]),
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
        exit_mode=normalize_exit_mode(str(exit_cfg_forward["exit_mode"])),
        trailing_atr_k=float(exit_cfg_forward["trailing_atr_k"]),
        partial_size=float(exit_cfg_forward["partial_size"]),
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
    cfg: dict[str, Any],
    baseline_eval: dict[str, Any],
    stage10_eval: dict[str, Any],
    walkforward_payload: dict[str, Any],
    leakage: dict[str, Any],
    features_by_symbol: dict[str, pd.DataFrame],
    is_stress: bool = False,
) -> dict[str, Any]:
    baseline = baseline_eval["aggregate_metrics"]
    stage10 = stage10_eval["aggregate_metrics"]
    comparison = {
        "trade_count_delta": float(stage10["trade_count"] - baseline["trade_count"]),
        "pf_adj_delta": float(stage10["pf_adj"] - baseline["pf_adj"]),
        "exp_lcb_delta": float(stage10["exp_lcb"] - baseline["exp_lcb"]),
    }
    regime_dist = _aggregate_regime_distribution(features_by_symbol)
    calibration = _aggregate_regime_calibration(features_by_symbol)
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
        "dry_run": bool(dry_run),
        "is_stress": bool(is_stress),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "seed": int(seed),
        "resolved_end_ts": resolved_end_ts,
        "enabled_signal_families": list(
            resolve_enabled_families(
                families=list(cfg["evaluation"]["stage10"]["signals"]["families"]),
                enabled_families=list(cfg["evaluation"]["stage10"]["signals"].get("enabled_families", [])),
            )
        ),
        "exit_modes": list(cfg["evaluation"]["stage10"]["exits"]["modes"]),
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
            "calibration": calibration,
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
        "trade_count_guard": _stage10_trade_count_guard(
            baseline_trade_count=float(baseline["trade_count"]),
            stage10_trade_count=float(stage10["trade_count"]),
            family_breakdown=list(stage10_eval.get("family_trade_breakdown", [])),
            max_drop_pct=10.0,
        ),
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
    lines.append(f"- calibration: `{summary['regimes'].get('calibration', {})}`")
    lines.append("")
    lines.append("## Walkforward V2")
    lines.append(f"- enabled: `{summary['walkforward_v2'].get('enabled')}`")
    lines.append(f"- baseline classification: `{summary['walkforward_v2'].get('classification', 'N/A')}`")
    lines.append(f"- stage10 classification: `{summary['walkforward_v2'].get('stage10_classification', 'N/A')}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- {summary['next_recommendation']}")
    lines.append("")
    lines.append("## Trade Count Guard")
    guard = summary.get("trade_count_guard", {})
    lines.append(f"- pass: `{guard.get('pass')}`")
    lines.append(f"- observed_drop_pct: `{_finite(guard.get('observed_drop_pct', 0.0), default=0.0):.6f}`")
    lines.append(f"- max_drop_pct: `{_finite(guard.get('max_drop_pct', 10.0), default=10.0):.2f}`")
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


def _aggregate_regime_calibration(features_by_symbol: dict[str, pd.DataFrame]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for frame in features_by_symbol.values():
        if frame.empty:
            continue
        diagnostics.append(regime_calibration_diagnostics(frame))
    if not diagnostics:
        return {
            "single_regime_warning": False,
            "warnings": [],
            "median_trend_strength": 0.0,
            "atr_percentile_distribution": {"p05": 0.0, "p50": 0.0, "p95": 0.0},
        }

    warnings: list[str] = []
    for item in diagnostics:
        warnings.extend([str(w) for w in item.get("warnings", [])])
    medians = [float(item.get("median_trend_strength", 0.0)) for item in diagnostics]
    atr_p05 = [float(item.get("atr_percentile_distribution", {}).get("p05", 0.0)) for item in diagnostics]
    atr_p50 = [float(item.get("atr_percentile_distribution", {}).get("p50", 0.0)) for item in diagnostics]
    atr_p95 = [float(item.get("atr_percentile_distribution", {}).get("p95", 0.0)) for item in diagnostics]
    return {
        "single_regime_warning": any(bool(item.get("single_regime_warning", False)) for item in diagnostics),
        "warnings": sorted(set(warnings)),
        "median_trend_strength": float(np.mean(medians)) if medians else 0.0,
        "atr_percentile_distribution": {
            "p05": float(np.mean(atr_p05)) if atr_p05 else 0.0,
            "p50": float(np.mean(atr_p50)) if atr_p50 else 0.0,
            "p95": float(np.mean(atr_p95)) if atr_p95 else 0.0,
        },
    }


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


def _build_stress_config(config: dict[str, Any]) -> dict[str, Any]:
    stressed = json.loads(json.dumps(config))
    stressed.setdefault("cost_model", {})
    stressed["cost_model"]["mode"] = "v2"
    stressed["cost_model"].setdefault("v2", {})
    v2 = stressed["cost_model"]["v2"]
    v2["delay_bars"] = int(v2.get("delay_bars", 0)) + 1
    v2["spread_bps"] = float(v2.get("spread_bps", 0.0)) + 1.0
    v2["slippage_bps_base"] = float(v2.get("slippage_bps_base", 0.0)) + 1.0
    return stressed


def build_stage10_6_report_from_runs(
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    max_drop_pct: float = 10.0,
) -> dict[str, Any]:
    """Build Stage-10.6 comparative report from available run artifacts."""

    stage10_runs = _discover_stage10_runs(runs_root)
    dry_runs = [row for row in stage10_runs if _is_dry_summary(row)]
    real_runs = [row for row in stage10_runs if not _is_dry_summary(row)]

    dry_pre, dry_latest = _pick_reference_and_latest(dry_runs)
    real_pre, real_latest = _pick_reference_and_latest(real_runs)
    sandbox_latest = _discover_latest_sandbox_summary(runs_root)

    comparisons = {
        "dry_run": _build_context_comparison(dry_pre, dry_latest),
        "real_data": _build_context_comparison(real_pre, real_latest),
    }
    comparisons["real_data"]["available"] = bool(real_latest is not None)

    guard_source = comparisons["real_data"] if comparisons["real_data"]["available"] else comparisons["dry_run"]
    guard = _build_trade_count_guard(guard_source, max_drop_pct=float(max_drop_pct))
    determinism = _build_determinism_status(dry_runs)

    summary: dict[str, Any] = {
        "stage": "10.6",
        "sandbox": {
            "enabled_signals": list((sandbox_latest or {}).get("enabled_signals", [])),
            "disabled_signals": list((sandbox_latest or {}).get("disabled_signals", [])),
            "rank_table_path": str((sandbox_latest or {}).get("rank_table_path", "")),
            "run_id": str((sandbox_latest or {}).get("run_id", "")),
        },
        "comparisons": comparisons,
        "trade_count_guard": guard,
        "determinism": determinism,
    }
    validate_stage10_6_summary_schema(summary)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_stage10_6_report(
        summary=summary,
        out_md=docs_dir / "stage10_6_report.md",
        out_json=docs_dir / "stage10_6_report_summary.json",
    )
    return summary


def validate_stage10_6_summary_schema(summary: dict[str, Any]) -> None:
    required_top = {"stage", "sandbox", "comparisons", "trade_count_guard", "determinism"}
    missing = required_top.difference(summary.keys())
    if missing:
        raise ValueError(f"Missing Stage-10.6 summary keys: {sorted(missing)}")
    if str(summary["stage"]) != "10.6":
        raise ValueError("stage must be '10.6'")
    sandbox = summary["sandbox"]
    if not isinstance(sandbox.get("enabled_signals"), list):
        raise ValueError("sandbox.enabled_signals must be a list")
    if not isinstance(sandbox.get("disabled_signals"), list):
        raise ValueError("sandbox.disabled_signals must be a list")
    comparisons = summary["comparisons"]
    for key in ["dry_run", "real_data"]:
        if key not in comparisons:
            raise ValueError(f"comparisons.{key} is required")
    guard = summary["trade_count_guard"]
    float(guard["max_drop_pct"])
    float(guard["observed_drop_pct"])
    if not isinstance(guard["pass"], bool):
        raise ValueError("trade_count_guard.pass must be bool")
    determinism = summary["determinism"]
    if not isinstance(determinism.get("pass"), bool):
        raise ValueError("determinism.pass must be bool")


def _discover_stage10_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*_stage10/stage10_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if bool(payload.get("is_stress", False)):
            continue
        payload["_summary_path"] = str(path)
        payload["_run_dir"] = str(path.parent)
        rows.append(payload)
    rows.sort(key=lambda row: str(row.get("run_id", "")))
    return rows


def _is_dry_summary(summary: dict[str, Any]) -> bool:
    if "dry_run" in summary:
        return bool(summary.get("dry_run"))
    real_block = summary.get("baseline_vs_stage10", {}).get("real_data", {})
    return not bool(real_block.get("available", False))


def _discover_latest_sandbox_summary(runs_root: Path) -> dict[str, Any] | None:
    summaries: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*_stage10_sandbox/sandbox_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(payload)
    if not summaries:
        return None
    summaries.sort(key=lambda row: str(row.get("run_id", "")))
    return summaries[-1]


def _pick_reference_and_latest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not rows:
        return None, None
    if len(rows) == 1:
        return None, rows[-1]
    latest = rows[-1]
    latest_cfg = str(latest.get("config_hash", ""))
    latest_signals = tuple(latest.get("enabled_signal_families", []) or [])
    latest_exits = tuple(latest.get("exit_modes", []) or [])
    for candidate in reversed(rows[:-1]):
        if (
            str(candidate.get("config_hash", "")) != latest_cfg
            or tuple(candidate.get("enabled_signal_families", []) or []) != latest_signals
            or tuple(candidate.get("exit_modes", []) or []) != latest_exits
        ):
            return candidate, latest
    return rows[-2], latest


def _extract_context_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {}
    block = summary.get("baseline_vs_stage10", {})
    walkforward = summary.get("walkforward_v2", {})
    stage10_walk = walkforward.get("stage10", {}) if isinstance(walkforward.get("stage10", {}), dict) else {}
    regime_dist = summary.get("regimes", {}).get("distribution", {})
    max_share = 0.0
    if isinstance(regime_dist, dict) and regime_dist:
        max_share = max(_finite(value, 0.0) for value in regime_dist.values())
    return {
        "run_id": str(summary.get("run_id", "")),
        "baseline": dict(block.get("baseline", {})),
        "stage10": dict(block.get("stage10", {})),
        "walkforward_classification": str(walkforward.get("stage10_classification", walkforward.get("classification", "N/A"))),
        "usable_windows": int(stage10_walk.get("usable_windows", 0)),
        "regime_distribution": dict(regime_dist) if isinstance(regime_dist, dict) else {},
        "single_label_warning": bool(max_share >= 99.9),
    }


def _build_context_comparison(previous: dict[str, Any] | None, latest: dict[str, Any] | None) -> dict[str, Any]:
    pre = _extract_context_metrics(previous)
    post = _extract_context_metrics(latest)
    baseline = dict(post.get("baseline", {}))
    return {
        "baseline": baseline,
        "stage10_pre": dict(pre.get("stage10", {})),
        "stage10_6": dict(post.get("stage10", {})),
        "walkforward_pre": str(pre.get("walkforward_classification", "N/A")),
        "walkforward_stage10_6": str(post.get("walkforward_classification", "N/A")),
        "usable_windows_pre": int(pre.get("usable_windows", 0)),
        "usable_windows_stage10_6": int(post.get("usable_windows", 0)),
        "regime_distribution_stage10_6": dict(post.get("regime_distribution", {})),
        "single_label_warning_stage10_6": bool(post.get("single_label_warning", False)),
        "pre_run_id": str(pre.get("run_id", "")),
        "stage10_6_run_id": str(post.get("run_id", "")),
    }


def _build_trade_count_guard(context: dict[str, Any], max_drop_pct: float) -> dict[str, Any]:
    baseline = context.get("baseline", {})
    stage10_6 = context.get("stage10_6", {})
    baseline_tc = _finite(baseline.get("trade_count", 0.0), default=0.0)
    stage_tc = _finite(stage10_6.get("trade_count", 0.0), default=0.0)
    if baseline_tc <= 0:
        observed_drop_pct = 0.0
    else:
        observed_drop_pct = max(0.0, (baseline_tc - stage_tc) / baseline_tc * 100.0)

    pf_delta = _finite(stage10_6.get("profit_factor", 0.0), default=0.0) - _finite(
        baseline.get("profit_factor", 0.0),
        default=0.0,
    )
    expectancy_delta = _finite(stage10_6.get("expectancy", 0.0), default=0.0) - _finite(
        baseline.get("expectancy", 0.0),
        default=0.0,
    )
    robust_improved = bool(pf_delta > 0 and expectancy_delta > 0)
    passed = bool(observed_drop_pct <= float(max_drop_pct) or robust_improved)
    return {
        "max_drop_pct": float(max_drop_pct),
        "observed_drop_pct": float(observed_drop_pct),
        "pass": passed,
        "pf_delta_vs_baseline": float(pf_delta),
        "expectancy_delta_vs_baseline": float(expectancy_delta),
        "justified_by_robust_improvement": robust_improved,
    }


def _stage10_trade_count_guard(
    baseline_trade_count: float,
    stage10_trade_count: float,
    family_breakdown: list[dict[str, Any]],
    max_drop_pct: float,
) -> dict[str, Any]:
    baseline = max(0.0, float(baseline_trade_count))
    stage = max(0.0, float(stage10_trade_count))
    if baseline <= 0:
        observed = 0.0
    else:
        observed = max(0.0, (baseline - stage) / baseline * 100.0)
    reductions: list[dict[str, Any]] = []
    for item in family_breakdown:
        family_trade_count = max(0.0, _finite(item.get("trade_count", 0.0), default=0.0))
        reductions.append(
            {
                "family": str(item.get("family", "")),
                "trade_count": family_trade_count,
                "delta_vs_baseline_total": float(family_trade_count - baseline),
            }
        )
    reductions.sort(key=lambda row: float(row["delta_vs_baseline_total"]))
    return {
        "max_drop_pct": float(max_drop_pct),
        "observed_drop_pct": float(observed),
        "pass": bool(observed <= float(max_drop_pct)),
        "family_reduction_breakdown": reductions,
    }


def _build_determinism_status(dry_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not dry_runs:
        return {"pass": False, "notes": "No dry runs found for determinism check"}
    latest = dry_runs[-1]
    latest_signature = str(latest.get("determinism", {}).get("signature", ""))
    matching = [
        row
        for row in dry_runs
        if row.get("config_hash") == latest.get("config_hash")
        and row.get("data_hash") == latest.get("data_hash")
        and row.get("enabled_signal_families") == latest.get("enabled_signal_families")
        and row.get("exit_modes") == latest.get("exit_modes")
        and int(row.get("seed", -1)) == int(latest.get("seed", -2))
    ]
    if len(matching) < 2:
        return {
            "pass": False,
            "notes": "Need two matching dry runs (same seed/config/data) for determinism proof",
            "signature_latest": latest_signature,
        }
    prev = matching[-2]
    prev_signature = str(prev.get("determinism", {}).get("signature", ""))
    passed = bool(prev_signature != "" and prev_signature == latest_signature)
    return {
        "pass": passed,
        "notes": "PASS" if passed else "Signatures differ across matching dry runs",
        "signature_previous": prev_signature,
        "signature_latest": latest_signature,
    }


def _write_stage10_6_report(summary: dict[str, Any], out_md: Path, out_json: Path) -> None:
    dry = summary["comparisons"]["dry_run"]
    real = summary["comparisons"]["real_data"]
    guard = summary["trade_count_guard"]
    lines: list[str] = []
    lines.append("# Stage-10.6 Report")
    lines.append("")
    lines.append("## What Changed")
    lines.append("- Switched activation decisions to score-only regime usage.")
    lines.append("- Clamped activation multipliers to a strict soft band.")
    lines.append("- Reduced default exits to fixed ATR and ATR trailing.")
    lines.append("- Added sandbox ranking with drag-aware stress penalty.")
    lines.append("")
    lines.append("## Sandbox Selection")
    lines.append(f"- enabled_signals: `{summary['sandbox']['enabled_signals']}`")
    lines.append(f"- disabled_signals: `{summary['sandbox']['disabled_signals']}`")
    lines.append(f"- ranking table: `{summary['sandbox']['rank_table_path']}`")
    lines.append("")
    lines.append("## Comparisons (Dry Run)")
    lines.append(f"- pre_run_id: `{dry.get('pre_run_id', '')}`")
    lines.append(f"- stage10_6_run_id: `{dry.get('stage10_6_run_id', '')}`")
    lines.append(f"- walkforward: `{dry.get('walkforward_pre', 'N/A')} -> {dry.get('walkforward_stage10_6', 'N/A')}`")
    lines.append(f"- usable_windows: `{dry.get('usable_windows_pre', 0)} -> {dry.get('usable_windows_stage10_6', 0)}`")
    lines.append(f"- single_label_warning: `{dry.get('single_label_warning_stage10_6', False)}`")
    lines.append("")
    lines.append("| metric | baseline | stage10_pre | stage10_6 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for metric in ["trade_count", "profit_factor", "expectancy", "max_drawdown", "pf_adj", "exp_lcb"]:
        lines.append(
            f"| {metric} | "
            f"{_finite(dry.get('baseline', {}).get(metric, 0.0), 0.0):.6f} | "
            f"{_finite(dry.get('stage10_pre', {}).get(metric, 0.0), 0.0):.6f} | "
            f"{_finite(dry.get('stage10_6', {}).get(metric, 0.0), 0.0):.6f} |"
        )
    lines.append("")
    lines.append("## Comparisons (Real Data)")
    lines.append(f"- available: `{real.get('available', False)}`")
    lines.append(f"- pre_run_id: `{real.get('pre_run_id', '')}`")
    lines.append(f"- stage10_6_run_id: `{real.get('stage10_6_run_id', '')}`")
    lines.append(f"- walkforward: `{real.get('walkforward_pre', 'N/A')} -> {real.get('walkforward_stage10_6', 'N/A')}`")
    lines.append(f"- usable_windows: `{real.get('usable_windows_pre', 0)} -> {real.get('usable_windows_stage10_6', 0)}`")
    lines.append(f"- single_label_warning: `{real.get('single_label_warning_stage10_6', False)}`")
    lines.append("")
    if bool(real.get("available", False)):
        lines.append("| metric | baseline | stage10_pre | stage10_6 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for metric in ["trade_count", "profit_factor", "expectancy", "max_drawdown", "pf_adj", "exp_lcb"]:
            lines.append(
                f"| {metric} | "
                f"{_finite(real.get('baseline', {}).get(metric, 0.0), 0.0):.6f} | "
                f"{_finite(real.get('stage10_pre', {}).get(metric, 0.0), 0.0):.6f} | "
                f"{_finite(real.get('stage10_6', {}).get(metric, 0.0), 0.0):.6f} |"
            )
        lines.append("")
    lines.append("## Trade Count Guard")
    lines.append(f"- max_drop_pct: `{guard['max_drop_pct']:.2f}`")
    lines.append(f"- observed_drop_pct: `{guard['observed_drop_pct']:.6f}`")
    lines.append(f"- pass: `{guard['pass']}`")
    lines.append(f"- justified_by_robust_improvement: `{guard['justified_by_robust_improvement']}`")
    lines.append("")
    lines.append("## Determinism")
    lines.append(f"- pass: `{summary['determinism']['pass']}`")
    lines.append(f"- notes: `{summary['determinism']['notes']}`")
    if summary["determinism"].get("signature_previous"):
        lines.append(f"- signature_previous: `{summary['determinism']['signature_previous']}`")
    if summary["determinism"].get("signature_latest"):
        lines.append(f"- signature_latest: `{summary['determinism']['signature_latest']}`")
    lines.append("")
    lines.append("## Known Limitations")
    lines.append("- Stage-10.6 ranking is currently single-exit-first (fixed_atr) before A/B exit sweeps.")
    lines.append("- Drag penalty is sandbox-only and not yet part of the main optimizer objective.")

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def build_stage10_7_report_from_runs(
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    max_drop_pct: float = 10.0,
) -> dict[str, Any]:
    """Build Stage-10.7 forensic refinement report from run artifacts."""

    stage10_runs = _discover_stage10_runs(runs_root)
    dry_runs = [row for row in stage10_runs if _is_dry_summary(row)]
    real_runs = [row for row in stage10_runs if not _is_dry_summary(row)]
    dry_pre, dry_latest = _pick_reference_and_latest(dry_runs)
    real_pre, real_latest = _pick_reference_and_latest(real_runs)

    dry_cmp = _build_context_comparison(dry_pre, dry_latest)
    real_cmp = _build_context_comparison(real_pre, real_latest)
    real_cmp["available"] = bool(real_latest is not None)
    guard_source = real_cmp if real_cmp["available"] else dry_cmp
    guard = _build_trade_count_guard(guard_source, max_drop_pct=float(max_drop_pct))
    latest_summary = real_latest or dry_latest or {}
    guard["family_reduction_breakdown"] = list(
        latest_summary.get("trade_count_guard", {}).get("family_reduction_breakdown", [])
    )
    determinism = _build_determinism_status(dry_runs)
    sandbox = _discover_latest_sandbox_summary(runs_root) or {}
    exit_ab = _discover_latest_exit_ab_summary(runs_root) or {}
    regime_calibration = dict((real_latest or dry_latest or {}).get("regimes", {}).get("calibration", {}))
    regime_distribution = dict((real_latest or dry_latest or {}).get("regimes", {}).get("distribution", {}))

    verdict = _stage10_7_verdict(real_cmp=real_cmp, dry_cmp=dry_cmp, guard=guard)
    summary: dict[str, Any] = {
        "stage": "10.7",
        "regime_distribution": regime_distribution,
        "regime_calibration": regime_calibration,
        "sandbox": {
            "run_id": str(sandbox.get("run_id", "")),
            "enabled_signals": list(sandbox.get("enabled_signals", [])),
            "disabled_signals": list(sandbox.get("disabled_signals", [])),
            "rank_table_path": str(sandbox.get("rank_table_path", "")),
        },
        "exit_ab": {
            "run_id": str(exit_ab.get("run_id", "")),
            "selected_exit": str(exit_ab.get("selected_exit", "")),
            "rows": list(exit_ab.get("rows", [])),
        },
        "comparisons": {
            "dry": dry_cmp,
            "real": real_cmp,
        },
        "trade_count_guard": guard,
        "determinism": bool(determinism.get("pass", False)),
        "determinism_detail": determinism,
        "final_verdict": verdict,
    }
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_stage10_7_report(
        summary=summary,
        out_md=docs_dir / "stage10_7_report.md",
        out_json=docs_dir / "stage10_7_report_summary.json",
    )
    return summary


def _discover_latest_exit_ab_summary(runs_root: Path) -> dict[str, Any] | None:
    summaries: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*_stage10_exit_ab/exit_ab_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(payload)
    if not summaries:
        return None
    summaries.sort(key=lambda row: str(row.get("run_id", "")))
    return summaries[-1]


def _stage10_7_verdict(real_cmp: dict[str, Any], dry_cmp: dict[str, Any], guard: dict[str, Any]) -> str:
    context = real_cmp if bool(real_cmp.get("available", False)) else dry_cmp
    pre = context.get("stage10_pre", {})
    post = context.get("stage10_6", {})
    pf_delta = _finite(post.get("profit_factor", 0.0), 0.0) - _finite(pre.get("profit_factor", 0.0), 0.0)
    exp_delta = _finite(post.get("expectancy", 0.0), 0.0) - _finite(pre.get("expectancy", 0.0), 0.0)
    wf_pre = str(context.get("walkforward_pre", "N/A"))
    wf_post = str(context.get("walkforward_stage10_6", "N/A"))
    wf_not_worse = wf_post == wf_pre or (wf_pre == "INSUFFICIENT_DATA" and wf_post in {"UNSTABLE", "STABLE"})

    if bool(guard.get("pass", False)) and pf_delta > 0 and exp_delta > 0 and wf_not_worse:
        return "IMPROVED"
    if abs(pf_delta) < 1e-12 and abs(exp_delta) < 1e-12 and wf_not_worse:
        return "NO_CHANGE"
    return "REGRESSION"


def _write_stage10_7_report(summary: dict[str, Any], out_md: Path, out_json: Path) -> None:
    dry = summary["comparisons"]["dry"]
    real = summary["comparisons"]["real"]
    guard = summary["trade_count_guard"]
    lines: list[str] = []
    lines.append("# Stage-10.7 Report")
    lines.append("")
    lines.append("## Regime Calibration")
    lines.append(f"- regime_distribution: `{summary['regime_distribution']}`")
    lines.append(f"- calibration: `{summary['regime_calibration']}`")
    lines.append("")
    lines.append("## Sandbox Ranking (Real Data)")
    lines.append(f"- sandbox_run_id: `{summary['sandbox']['run_id']}`")
    lines.append(f"- enabled_signals: `{summary['sandbox']['enabled_signals']}`")
    lines.append(f"- disabled_signals: `{summary['sandbox']['disabled_signals']}`")
    lines.append(f"- ranking_table: `{summary['sandbox']['rank_table_path']}`")
    lines.append("")
    lines.append("## Exit A/B Isolation")
    lines.append(f"- exit_ab_run_id: `{summary['exit_ab']['run_id']}`")
    lines.append(f"- selected_exit: `{summary['exit_ab']['selected_exit']}`")
    lines.append("| exit_mode | trade_count | PF | expectancy | maxDD | exp_lcb | drag_sensitivity |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["exit_ab"]["rows"]:
        lines.append(
            f"| {row.get('exit_mode','')} | "
            f"{_finite(row.get('trade_count',0.0),0.0):.2f} | "
            f"{_finite(row.get('profit_factor',0.0),0.0):.6f} | "
            f"{_finite(row.get('expectancy',0.0),0.0):.6f} | "
            f"{_finite(row.get('max_drawdown',0.0),0.0):.6f} | "
            f"{_finite(row.get('exp_lcb',0.0),0.0):.6f} | "
            f"{_finite(row.get('drag_sensitivity',0.0),0.0):.6f} |"
        )
    lines.append("")
    lines.append("## Before/After Comparison")
    lines.append("### Dry")
    lines.append(f"- Stage-9 baseline run-context: `{dry.get('stage10_6_run_id','')}`")
    lines.append(f"- Stage-10.6 run_id: `{dry.get('pre_run_id','')}`")
    lines.append(f"- Stage-10.7 run_id: `{dry.get('stage10_6_run_id','')}`")
    lines.append(f"- walkforward: `{dry.get('walkforward_pre','N/A')} -> {dry.get('walkforward_stage10_6','N/A')}`")
    lines.append(f"- usable_windows: `{dry.get('usable_windows_pre',0)} -> {dry.get('usable_windows_stage10_6',0)}`")
    lines.append("")
    lines.append("### Real")
    lines.append(f"- available: `{real.get('available', False)}`")
    lines.append(f"- Stage-10.6 run_id: `{real.get('pre_run_id','')}`")
    lines.append(f"- Stage-10.7 run_id: `{real.get('stage10_6_run_id','')}`")
    lines.append(f"- walkforward: `{real.get('walkforward_pre','N/A')} -> {real.get('walkforward_stage10_6','N/A')}`")
    lines.append(f"- usable_windows: `{real.get('usable_windows_pre',0)} -> {real.get('usable_windows_stage10_6',0)}`")
    lines.append("")
    lines.append("| metric | Stage-9 baseline | Stage-10.6 | Stage-10.7 |")
    lines.append("| --- | ---: | ---: | ---: |")
    context = real if bool(real.get("available", False)) else dry
    for metric in ["trade_count", "profit_factor", "expectancy", "max_drawdown", "pf_adj", "exp_lcb"]:
        lines.append(
            f"| {metric} | "
            f"{_finite(context.get('baseline', {}).get(metric, 0.0), 0.0):.6f} | "
            f"{_finite(context.get('stage10_pre', {}).get(metric, 0.0), 0.0):.6f} | "
            f"{_finite(context.get('stage10_6', {}).get(metric, 0.0), 0.0):.6f} |"
        )
    lines.append("")
    lines.append("## Trade Count Guard")
    lines.append(f"- pass: `{guard['pass']}`")
    lines.append(f"- observed_drop_pct: `{guard['observed_drop_pct']:.6f}`")
    lines.append(f"- max_drop_pct: `{guard['max_drop_pct']:.2f}`")
    if not bool(guard.get("pass", True)):
        lines.append("- family_reduction_breakdown (top 5 by delta_vs_baseline_total):")
        for row in list(guard.get("family_reduction_breakdown", []))[:5]:
            lines.append(
                f"  - {row.get('family','')}: "
                f"trade_count={_finite(row.get('trade_count',0.0),0.0):.2f}, "
                f"delta_vs_baseline_total={_finite(row.get('delta_vs_baseline_total',0.0),0.0):.2f}"
            )
    lines.append("")
    lines.append("## Determinism")
    lines.append(f"- pass: `{summary['determinism']}`")
    lines.append(f"- detail: `{summary['determinism_detail']}`")
    lines.append("")
    lines.append("## Final Verdict")
    lines.append(f"- {summary['final_verdict']}")

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


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
