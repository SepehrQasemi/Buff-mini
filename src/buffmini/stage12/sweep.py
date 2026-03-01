"""Stage-12 full price-family sweep with walk-forward and Monte Carlo robustness checks."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, stage06_strategies
from buffmini.config import compute_config_hash, validate_config
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.cache import ohlcv_data_hash
from buffmini.portfolio.monte_carlo import simulate_equity_paths, summarize_mc
from buffmini.stage10.evaluate import _build_features
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.stage10.signals import SIGNAL_FAMILIES, generate_signal_family
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.walkforward_v2 import aggregate_windows, build_windows


_COST_LEVEL_ORDER = {"low": 0, "realistic": 1, "high": 2}
_SUMMARY_STAGE = "12"


@dataclass(frozen=True)
class StrategyVariant:
    strategy_key: str
    strategy_name: str
    source: str
    signal_builder: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ExitVariant:
    exit_type: str
    engine_exit_mode: str
    stop_atr_multiple: float
    take_profit_atr_multiple: float
    trailing_atr_k: float
    max_hold_bars: int


def run_stage12_sweep(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = False,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Run full Stage-12 matrix sweep and write run/docs artifacts."""

    cfg = deepcopy(config)
    validate_config(cfg)
    stage12_cfg = cfg.get("evaluation", {}).get("stage12", {})
    if not isinstance(stage12_cfg, dict):
        raise ValueError("evaluation.stage12 must be configured")

    resolved_symbols = list(symbols or stage12_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"]))
    resolved_timeframes = list(timeframes or stage12_cfg.get("timeframes", ["15m", "30m", "1h", "2h", "4h", "1d"]))

    strategies = _resolve_strategies(cfg=cfg, stage12_cfg=stage12_cfg)
    exits = _resolve_exits(cfg=cfg, stage12_cfg=stage12_cfg)
    cost_scenarios = _resolve_cost_scenarios(cfg=cfg, stage12_cfg=stage12_cfg)
    if not strategies:
        raise ValueError("No Stage-12 strategies resolved")
    if not exits:
        raise ValueError("No Stage-12 exit variants resolved")
    if not cost_scenarios:
        raise ValueError("No Stage-12 cost scenarios resolved")

    run_started = time.perf_counter()
    runtime_by_tf: dict[str, float] = {}
    feature_hashes: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    trade_pnls_by_combo: dict[str, np.ndarray] = {}
    window_details_rows: list[dict[str, Any]] = []
    min_usable_windows_valid = int(stage12_cfg.get("min_usable_windows_valid", 3))

    for timeframe in resolved_timeframes:
        tf_started = time.perf_counter()
        cfg_tf = _config_for_timeframe(cfg=cfg, timeframe=str(timeframe))
        features_by_symbol = _build_features(
            config=cfg_tf,
            symbols=resolved_symbols,
            timeframe=str(timeframe),
            dry_run=bool(dry_run),
            seed=int(seed),
            data_dir=data_dir,
            derived_dir=derived_dir,
        )
        feature_hashes[str(timeframe)] = {
            str(symbol): ohlcv_data_hash(frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]])
            for symbol, frame in features_by_symbol.items()
            if not frame.empty
        }
        for symbol in resolved_symbols:
            frame = features_by_symbol.get(symbol)
            if frame is None or frame.empty:
                continue
            frame_sorted = frame.copy().sort_values("timestamp").reset_index(drop=True)
            frame_sorted.attrs["timeframe"] = str(timeframe)
            windows = _build_windows_for_frame(frame_sorted, cfg_tf)
            for strategy in strategies:
                for exit_variant in exits:
                    for cost_level, cost_model_cfg in cost_scenarios.items():
                        frame_sorted.attrs["cost_level"] = cost_level
                        combo_key = _combo_key(
                            symbol=symbol,
                            timeframe=str(timeframe),
                            strategy=strategy.strategy_key,
                            exit_type=exit_variant.exit_type,
                            cost_level=cost_level,
                        )
                        combo_eval = _evaluate_combo(
                            frame=frame_sorted,
                            symbol=symbol,
                            strategy=strategy,
                            exit_variant=exit_variant,
                            cost_model_cfg=cost_model_cfg,
                            cfg=cfg_tf,
                            windows=windows,
                            min_usable_windows_valid=min_usable_windows_valid,
                        )
                        combo_eval["combo_key"] = combo_key
                        rows.append(combo_eval)
                        trade_pnls_by_combo[combo_key] = combo_eval.pop("_trade_pnls")
                        for window_row in combo_eval.pop("_window_rows"):
                            window_details_rows.append(
                                {
                                    "combo_key": combo_key,
                                    "symbol": symbol,
                                    "timeframe": str(timeframe),
                                    "strategy": strategy.strategy_name,
                                    "exit_type": exit_variant.exit_type,
                                    "cost_level": cost_level,
                                    **window_row,
                                }
                            )
        runtime_by_tf[str(timeframe)] = float(time.perf_counter() - tf_started)

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        raise RuntimeError("Stage-12 produced no combinations")
    leaderboard = _apply_cost_sensitivity_and_robust_score(leaderboard=leaderboard, stage12_cfg=stage12_cfg)
    leaderboard = _apply_monte_carlo(
        leaderboard=leaderboard,
        trade_pnls_by_combo=trade_pnls_by_combo,
        stage12_cfg=stage12_cfg,
        seed=int(seed),
    )

    leaderboard["stability_rank"] = leaderboard["stability_classification"].map(
        {"STABLE": 0, "UNSTABLE": 1, "INVALID": 2, "INSUFFICIENT_DATA": 3}
    ).fillna(4)
    leaderboard = leaderboard.sort_values(
        ["exp_lcb", "stability_rank", "cost_sensitivity_slope", "symbol", "timeframe", "strategy", "exit_type", "cost_level"],
        ascending=[False, True, True, True, True, True, True, True],
    ).reset_index(drop=True)
    leaderboard = leaderboard.drop(columns=["stability_rank"], errors="ignore")

    total_runtime = float(time.perf_counter() - run_started)
    valid_combos = int((leaderboard["is_valid"] == True).sum())  # noqa: E712
    total_combos = int(len(leaderboard))
    verdict = _final_verdict(leaderboard)
    top_robust = _top_robust_rows(leaderboard, limit=10)

    summary_payload = {
        "stage": _SUMMARY_STAGE,
        "symbols": resolved_symbols,
        "timeframes": resolved_timeframes,
        "total_combinations": total_combos,
        "valid_combinations": valid_combos,
        "top_robust": top_robust,
        "verdict": verdict,
        "runtime_seconds": total_runtime,
        "pytest_pass_count": None,
        "seed": int(seed),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(feature_hashes, length=16),
        "runtime_by_timeframe": runtime_by_tf,
        "strategy_count": len(strategies),
        "exit_count": len(exits),
        "cost_levels": list(cost_scenarios.keys()),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(summary_payload, length=12)}_stage12"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
    pd.DataFrame(window_details_rows).to_csv(run_dir / "window_metrics.csv", index=False)
    pd.DataFrame(
        [{"timeframe": tf, "runtime_seconds": seconds} for tf, seconds in runtime_by_tf.items()]
    ).to_csv(run_dir / "runtime_by_timeframe.csv", index=False)

    summary_payload["run_id"] = run_id
    (run_dir / "stage12_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage12_report.md"
    report_json = docs_dir / "stage12_report_summary.json"
    _write_docs_report(
        leaderboard=leaderboard,
        summary=summary_payload,
        report_md=report_md,
        report_json=report_json,
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "summary": summary_payload,
        "leaderboard": leaderboard,
        "report_md": report_md,
        "report_json": report_json,
    }


def validate_stage12_summary_schema(payload: dict[str, Any]) -> None:
    required = {
        "stage",
        "symbols",
        "timeframes",
        "total_combinations",
        "valid_combinations",
        "top_robust",
        "verdict",
        "runtime_seconds",
        "pytest_pass_count",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Missing Stage-12 summary keys: {sorted(missing)}")
    if str(payload["stage"]) != _SUMMARY_STAGE:
        raise ValueError("stage must be '12'")
    if str(payload["verdict"]) not in {"STRONG EDGE FOUND", "WEAK EDGE", "NO ROBUST EDGE"}:
        raise ValueError("verdict must be STRONG EDGE FOUND / WEAK EDGE / NO ROBUST EDGE")
    if int(payload["total_combinations"]) < 0:
        raise ValueError("total_combinations must be >= 0")
    if int(payload["valid_combinations"]) < 0:
        raise ValueError("valid_combinations must be >= 0")
    if float(payload["runtime_seconds"]) < 0:
        raise ValueError("runtime_seconds must be >= 0")
    if not isinstance(payload["top_robust"], list):
        raise ValueError("top_robust must be a list")


def _resolve_strategies(cfg: dict[str, Any], stage12_cfg: dict[str, Any]) -> list[StrategyVariant]:
    variants: list[StrategyVariant] = []
    if bool(stage12_cfg.get("include_stage06_baselines", True)):
        for spec in stage06_strategies():
            strategy_key = str(spec.name).replace(" ", "_")
            variants.append(
                StrategyVariant(
                    strategy_key=strategy_key,
                    strategy_name=str(spec.name),
                    source="stage06",
                    signal_builder=lambda frame, s=spec: generate_signals(frame, strategy=s, gating_mode="none"),
                )
            )
    if bool(stage12_cfg.get("include_stage10_families", True)):
        defaults = cfg.get("evaluation", {}).get("stage10", {}).get("signals", {}).get("defaults", {})
        families = cfg.get("evaluation", {}).get("stage10", {}).get("signals", {}).get("families", list(SIGNAL_FAMILIES))
        for family in families:
            family_name = str(family)
            if family_name not in SIGNAL_FAMILIES:
                continue
            params = dict(defaults.get(family_name, {}))
            variants.append(
                StrategyVariant(
                    strategy_key=family_name,
                    strategy_name=_family_display_name(family_name),
                    source="stage10",
                    signal_builder=lambda frame, fam=family_name, par=params: generate_signal_family(frame, family=fam, params=par)[
                        "signal"
                    ].astype(int),
                )
            )
    dedup: dict[str, StrategyVariant] = {}
    for item in variants:
        dedup[item.strategy_key] = item
    return [dedup[key] for key in sorted(dedup)]


def _resolve_exits(cfg: dict[str, Any], stage12_cfg: dict[str, Any]) -> list[ExitVariant]:
    eval_cfg = cfg.get("evaluation", {}).get("stage10", {}).get("evaluation", {})
    exits_cfg = cfg.get("evaluation", {}).get("stage10", {}).get("exits", {})
    base_stop = float(eval_cfg.get("stop_atr_multiple", 1.5))
    base_tp = float(eval_cfg.get("take_profit_atr_multiple", 3.0))
    base_max_hold = int(eval_cfg.get("max_hold_bars", 24))
    trailing_k = float(exits_cfg.get("trailing_atr_k", 1.5))
    variants = []
    for name in stage12_cfg.get("exits", {}).get("variants", []):
        key = str(name)
        if key == "fixed_atr":
            variants.append(
                ExitVariant(
                    exit_type="fixed_atr",
                    engine_exit_mode=normalize_exit_mode("fixed_atr"),
                    stop_atr_multiple=base_stop,
                    take_profit_atr_multiple=base_tp,
                    trailing_atr_k=trailing_k,
                    max_hold_bars=base_max_hold,
                )
            )
        elif key == "structure_trailing":
            variants.append(
                ExitVariant(
                    exit_type="structure_trailing",
                    engine_exit_mode=normalize_exit_mode("atr_trailing"),
                    stop_atr_multiple=base_stop,
                    take_profit_atr_multiple=base_tp,
                    trailing_atr_k=trailing_k,
                    max_hold_bars=base_max_hold,
                )
            )
        elif key == "time_based":
            variants.append(
                ExitVariant(
                    exit_type="time_based",
                    engine_exit_mode=normalize_exit_mode("fixed_atr"),
                    stop_atr_multiple=float(stage12_cfg["exits"]["time_based_stop_atr_multiple"]),
                    take_profit_atr_multiple=float(stage12_cfg["exits"]["time_based_take_profit_atr_multiple"]),
                    trailing_atr_k=trailing_k,
                    max_hold_bars=base_max_hold,
                )
            )
    return variants


def _resolve_cost_scenarios(cfg: dict[str, Any], stage12_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_cfg = deepcopy(cfg.get("cost_model", {}))
    base_cfg["mode"] = "v2"
    base_cfg.setdefault("v2", {})
    scenarios = stage12_cfg.get("cost_scenarios", {})
    out: dict[str, dict[str, Any]] = {}
    for level in ("low", "realistic", "high"):
        if level == "realistic":
            out[level] = deepcopy(base_cfg)
            continue
        payload = deepcopy(base_cfg)
        overrides = dict(scenarios.get(level, {}))
        payload["v2"]["slippage_bps_base"] = float(overrides.get("slippage_bps_base", payload["v2"].get("slippage_bps_base", 0.5)))
        payload["v2"]["slippage_bps_vol_mult"] = float(
            overrides.get("slippage_bps_vol_mult", payload["v2"].get("slippage_bps_vol_mult", 2.0))
        )
        payload["v2"]["spread_bps"] = float(overrides.get("spread_bps", payload["v2"].get("spread_bps", 0.5)))
        payload["v2"]["delay_bars"] = int(overrides.get("delay_bars", payload["v2"].get("delay_bars", 0)))
        out[level] = payload
    return out


def _config_for_timeframe(cfg: dict[str, Any], timeframe: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    universe = out.setdefault("universe", {})
    base_tf = str(universe.get("base_timeframe", timeframe))
    universe["timeframe"] = str(timeframe)
    universe["operational_timeframe"] = str(timeframe)
    if base_tf == "1m" and str(timeframe) != "1m":
        out.setdefault("data", {})["resample_source"] = "base"
    else:
        out.setdefault("data", {}).setdefault("resample_source", "direct")
    return out


def _build_windows_for_frame(frame: pd.DataFrame, cfg: dict[str, Any]) -> list[Any]:
    wf_cfg = cfg.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {})
    if frame.empty:
        return []
    return build_windows(
        start_ts=frame["timestamp"].iloc[0],
        end_ts=frame["timestamp"].iloc[-1],
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )


def _evaluate_combo(
    *,
    frame: pd.DataFrame,
    symbol: str,
    strategy: StrategyVariant,
    exit_variant: ExitVariant,
    cost_model_cfg: dict[str, Any],
    cfg: dict[str, Any],
    windows: list[Any],
    min_usable_windows_valid: int,
) -> dict[str, Any]:
    signal_series = strategy.signal_builder(frame)
    backtest_result = _run_backtest_with_signal(
        frame=frame,
        signal=signal_series,
        symbol=symbol,
        strategy_name=strategy.strategy_name,
        exit_variant=exit_variant,
        cfg=cfg,
        cost_model_cfg=cost_model_cfg,
    )
    metrics = _metrics_from_backtest(backtest_result=backtest_result, frame=frame)
    window_rows, wf_summary = _evaluate_walkforward_combo(
        frame=frame,
        symbol=symbol,
        strategy=strategy,
        exit_variant=exit_variant,
        cfg=cfg,
        cost_model_cfg=cost_model_cfg,
        windows=windows,
    )
    stability_classification = str(wf_summary.get("classification", "INSUFFICIENT_DATA"))
    usable_windows = int(wf_summary.get("usable_windows", 0))
    if usable_windows < int(min_usable_windows_valid):
        stability_effective = "INVALID"
        is_valid = False
    else:
        stability_effective = stability_classification
        is_valid = stability_classification != "INSUFFICIENT_DATA"
    output = {
        "symbol": str(symbol),
        "timeframe": str(frame.attrs.get("timeframe", "unknown")),
        "strategy": strategy.strategy_name,
        "strategy_key": strategy.strategy_key,
        "strategy_source": strategy.source,
        "exit_type": exit_variant.exit_type,
        "cost_level": str(frame.attrs.get("cost_level", "unknown")),
        "PF": float(metrics["profit_factor"]),
        "exp_lcb": float(metrics["exp_lcb"]),
        "expectancy": float(metrics["expectancy"]),
        "trades_per_month": float(metrics["trades_per_month"]),
        "maxDD": float(metrics["max_drawdown"]),
        "trade_count": float(metrics["trade_count"]),
        "stability_classification": stability_effective,
        "usable_windows": usable_windows,
        "is_valid": bool(is_valid),
        "walkforward_classification_raw": stability_classification,
        "cost_sensitivity_slope": 0.0,
        "robust_score": 0.0,
        "MC_p_ruin": math.nan,
        "MC_p_return_negative": math.nan,
        "MC_maxDD_p95": math.nan,
        "MC_expected_log_growth": math.nan,
        "_trade_pnls": pd.to_numeric(backtest_result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .to_numpy(dtype=float),
        "_window_rows": window_rows,
    }
    return output


def _run_backtest_with_signal(
    *,
    frame: pd.DataFrame,
    signal: pd.Series,
    symbol: str,
    strategy_name: str,
    exit_variant: ExitVariant,
    cfg: dict[str, Any],
    cost_model_cfg: dict[str, Any],
) -> Any:
    eval_cfg = cfg.get("evaluation", {}).get("stage10", {}).get("evaluation", {})
    costs = cfg.get("costs", {})
    work = frame.copy()
    work["signal"] = pd.to_numeric(signal, errors="coerce").fillna(0).astype(int)
    return run_backtest(
        frame=work,
        strategy_name=str(strategy_name),
        symbol=str(symbol),
        signal_col="signal",
        exit_mode=str(exit_variant.engine_exit_mode),
        trailing_atr_k=float(exit_variant.trailing_atr_k),
        max_hold_bars=int(exit_variant.max_hold_bars),
        stop_atr_multiple=float(exit_variant.stop_atr_multiple),
        take_profit_atr_multiple=float(exit_variant.take_profit_atr_multiple),
        round_trip_cost_pct=float(costs.get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(costs.get("slippage_pct", 0.0005)),
        initial_capital=float(eval_cfg.get("initial_capital", 10000.0)),
        cost_model_cfg=cost_model_cfg,
    )


def _evaluate_walkforward_combo(
    *,
    frame: pd.DataFrame,
    symbol: str,
    strategy: StrategyVariant,
    exit_variant: ExitVariant,
    cfg: dict[str, Any],
    cost_model_cfg: dict[str, Any],
    windows: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not windows:
        return [], {"classification": "INSUFFICIENT_DATA", "usable_windows": 0}
    wf_cfg = cfg.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {})
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    for window in windows:
        holdout = frame.loc[(ts >= window.holdout_start) & (ts < window.holdout_end)].copy().reset_index(drop=True)
        forward = frame.loc[(ts >= window.forward_start) & (ts < window.forward_end)].copy().reset_index(drop=True)
        hold_signal = strategy.signal_builder(holdout) if not holdout.empty else pd.Series(dtype=int)
        fwd_signal = strategy.signal_builder(forward) if not forward.empty else pd.Series(dtype=int)
        hold_result = _run_backtest_with_signal(
            frame=holdout,
            signal=hold_signal,
            symbol=symbol,
            strategy_name=strategy.strategy_name,
            exit_variant=exit_variant,
            cfg=cfg,
            cost_model_cfg=cost_model_cfg,
        ) if not holdout.empty else None
        fwd_result = _run_backtest_with_signal(
            frame=forward,
            signal=fwd_signal,
            symbol=symbol,
            strategy_name=strategy.strategy_name,
            exit_variant=exit_variant,
            cfg=cfg,
            cost_model_cfg=cost_model_cfg,
        ) if not forward.empty else None
        hold_metrics = _window_metrics(result=hold_result, frame=holdout)
        fwd_metrics = _window_metrics(result=fwd_result, frame=forward)
        usable, reasons = _usable_window(
            forward_metrics=fwd_metrics,
            min_trades=float(wf_cfg.get("min_trades", 10)),
            min_exposure=float(wf_cfg.get("min_exposure", 0.01)),
        )
        rows.append(
            {
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
                "exclude_reasons": ";".join(reasons),
                "holdout_expectancy": float(hold_metrics["expectancy"]),
                "holdout_profit_factor": float(hold_metrics["profit_factor"]),
                "holdout_max_drawdown": float(hold_metrics["max_drawdown"]),
                "holdout_return_pct": float(hold_metrics["return_pct"]),
                "holdout_trade_count": int(hold_metrics["trade_count"]),
                "holdout_exposure_ratio": float(hold_metrics["exposure_ratio"]),
                "forward_expectancy": float(fwd_metrics["expectancy"]),
                "forward_profit_factor": float(fwd_metrics["profit_factor"]),
                "forward_max_drawdown": float(fwd_metrics["max_drawdown"]),
                "forward_return_pct": float(fwd_metrics["return_pct"]),
                "forward_trade_count": int(fwd_metrics["trade_count"]),
                "forward_exposure_ratio": float(fwd_metrics["exposure_ratio"]),
            }
        )
    return rows, aggregate_windows(rows, cfg=cfg)


def _window_metrics(result: Any, frame: pd.DataFrame) -> dict[str, float]:
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
    trades = result.trades.copy()
    equity = result.equity_curve.copy()
    trade_count = float(result.metrics.get("trade_count", 0.0))
    exposure_ratio = 0.0
    if not trades.empty and "bars_held" in trades.columns:
        exposure_ratio = float(pd.to_numeric(trades["bars_held"], errors="coerce").fillna(0.0).sum() / max(1, len(frame)))
    if not equity.empty:
        initial = float(equity["equity"].iloc[0])
        final = float(equity["equity"].iloc[-1])
        return_pct = float((final / initial) - 1.0) if initial != 0 else 0.0
    else:
        return_pct = 0.0
    payload = {
        "expectancy": _finite(result.metrics.get("expectancy", 0.0), default=0.0),
        "profit_factor": _finite(result.metrics.get("profit_factor", 0.0), default=0.0),
        "max_drawdown": _finite(result.metrics.get("max_drawdown", 0.0), default=0.0),
        "return_pct": _finite(return_pct, default=0.0),
        "trade_count": _finite(trade_count, default=0.0),
        "exposure_ratio": _finite(exposure_ratio, default=0.0),
    }
    payload["finite"] = all(np.isfinite(float(value)) for value in payload.values())
    return payload


def _usable_window(forward_metrics: dict[str, float], min_trades: float, min_exposure: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(forward_metrics.get("finite", False)):
        reasons.append("non_finite_metrics")
    if float(forward_metrics.get("trade_count", 0.0)) < float(min_trades):
        reasons.append("min_trades")
    if float(forward_metrics.get("exposure_ratio", 0.0)) < float(min_exposure):
        reasons.append("min_exposure")
    return len(reasons) == 0, reasons


def _metrics_from_backtest(backtest_result: Any, frame: pd.DataFrame) -> dict[str, float]:
    metrics = dict(backtest_result.metrics)
    trade_count = _finite(metrics.get("trade_count", 0.0), default=0.0)
    expectancy = _finite(metrics.get("expectancy", 0.0), default=0.0)
    profit_factor = _finite(metrics.get("profit_factor", 0.0), default=0.0, clip=10.0)
    max_drawdown = _finite(metrics.get("max_drawdown", 0.0), default=0.0)
    pnl_values = pd.to_numeric(backtest_result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    exp_lcb = _exp_lcb(pnl_values)
    months = _estimate_months(frame)
    trades_per_month = float(trade_count / months) if months > 0 else 0.0
    return {
        "trade_count": trade_count,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "exp_lcb": exp_lcb,
        "trades_per_month": trades_per_month,
    }


def _estimate_months(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    days = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(days / 30.0, 1e-6)


def _exp_lcb(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean
    std = float(np.std(values, ddof=0))
    return float(mean - std / math.sqrt(float(values.size)))


def _apply_cost_sensitivity_and_robust_score(leaderboard: pd.DataFrame, stage12_cfg: dict[str, Any]) -> pd.DataFrame:
    frame = leaderboard.copy()
    penalties = stage12_cfg.get("robustness", {}).get("instability_penalty", {})
    penalty_weight = float(stage12_cfg.get("robustness", {}).get("cost_sensitivity_penalty_weight", 1.0))
    group_cols = ["symbol", "timeframe", "strategy", "strategy_key", "strategy_source", "exit_type"]
    for _, group in frame.groupby(group_cols, sort=True):
        idx = list(group.index)
        ranked = group.assign(_rank=group["cost_level"].map(_COST_LEVEL_ORDER)).dropna(subset=["_rank"]).sort_values("_rank")
        x = ranked["_rank"].to_numpy(dtype=float)
        y = ranked["exp_lcb"].to_numpy(dtype=float)
        if len(x) >= 2:
            slope = float(abs(np.polyfit(x, y, 1)[0]))
        else:
            slope = 0.0
        median_exp_lcb = float(np.median(y)) if len(y) else 0.0
        instability = max(
            float(penalties.get(str(label), penalties.get("INSUFFICIENT_DATA", 1.0)))
            for label in group["stability_classification"].tolist()
        )
        robust_score = float(median_exp_lcb - (penalty_weight * slope) - instability)
        frame.loc[idx, "cost_sensitivity_slope"] = slope
        frame.loc[idx, "robust_score"] = robust_score
    return frame


def _apply_monte_carlo(
    *,
    leaderboard: pd.DataFrame,
    trade_pnls_by_combo: dict[str, np.ndarray],
    stage12_cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    mc_cfg = stage12_cfg.get("monte_carlo", {})
    if not bool(mc_cfg.get("enabled", True)):
        return leaderboard
    frame = leaderboard.copy()
    valid = frame.loc[frame["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return frame
    valid = valid.sort_values("exp_lcb", ascending=False).reset_index(drop=True)
    top_pct = float(mc_cfg.get("top_pct", 0.2))
    top_n = max(1, int(math.ceil(len(valid) * top_pct)))
    selected_keys = valid.iloc[:top_n]["combo_key"].tolist()
    for idx, combo_key in enumerate(selected_keys):
        pnls = np.asarray(trade_pnls_by_combo.get(combo_key, np.asarray([], dtype=float)), dtype=float)
        if pnls.size == 0:
            continue
        paths = simulate_equity_paths(
            trade_pnls=pnls,
            n_paths=int(mc_cfg.get("n_paths", 5000)),
            method=str(mc_cfg.get("bootstrap", "block")),
            seed=int(seed) + int(idx),
            initial_equity=float(mc_cfg.get("initial_equity", 10000.0)),
            leverage=1.0,
            block_size_trades=int(mc_cfg.get("block_size_trades", 10)),
        )
        summary = summarize_mc(
            paths_results=paths,
            initial_equity=float(mc_cfg.get("initial_equity", 10000.0)),
            ruin_dd_threshold=float(mc_cfg.get("ruin_dd_threshold", 0.5)),
        )
        final_equity = pd.to_numeric(paths["final_equity"], errors="coerce").dropna().to_numpy(dtype=float)
        expected_log_growth = float(
            np.log(np.clip(final_equity, 1e-12, None) / float(mc_cfg.get("initial_equity", 10000.0))).mean()
        )
        frame.loc[frame["combo_key"] == combo_key, "MC_p_ruin"] = float(summary["tail_probabilities"]["p_ruin"])
        frame.loc[frame["combo_key"] == combo_key, "MC_p_return_negative"] = float(summary["tail_probabilities"]["p_return_lt_0"])
        frame.loc[frame["combo_key"] == combo_key, "MC_maxDD_p95"] = float(summary["max_drawdown"]["p95"])
        frame.loc[frame["combo_key"] == combo_key, "MC_expected_log_growth"] = expected_log_growth
    return frame


def _final_verdict(leaderboard: pd.DataFrame) -> str:
    valid = leaderboard.loc[leaderboard["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return "NO ROBUST EDGE"
    stable = valid.loc[valid["stability_classification"] == "STABLE"].copy()
    if stable.empty:
        if float(valid["robust_score"].max()) > 0:
            return "WEAK EDGE"
        return "NO ROBUST EDGE"
    top = stable.sort_values("robust_score", ascending=False).iloc[0]
    p_ruin = float(top.get("MC_p_ruin", math.nan))
    if float(top.get("robust_score", 0.0)) > 0 and (math.isnan(p_ruin) or p_ruin < 0.01):
        return "STRONG EDGE FOUND"
    if float(top.get("robust_score", 0.0)) > 0:
        return "WEAK EDGE"
    return "NO ROBUST EDGE"


def _top_robust_rows(leaderboard: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    cols = [
        "symbol",
        "timeframe",
        "strategy",
        "exit_type",
        "cost_level",
        "exp_lcb",
        "PF",
        "expectancy",
        "robust_score",
        "cost_sensitivity_slope",
        "stability_classification",
        "MC_p_ruin",
        "MC_p_return_negative",
    ]
    top = leaderboard.sort_values(["robust_score", "exp_lcb"], ascending=[False, False]).head(limit)
    records = top.loc[:, cols].to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for row in records:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                safe_row[key] = None
            else:
                safe_row[key] = value
        out.append(safe_row)
    return out


def _family_display_name(family: str) -> str:
    mapping = {
        "BreakoutRetest": "Breakout Retest",
        "MA_SlopePullback": "MA SlopePullback",
        "VolCompressionBreakout": "Vol Compression Breakout",
        "BollingerSnapBack": "Bollinger SnapBack",
        "ATR_DistanceRevert": "ATR Distance Revert",
        "RangeFade": "Range Fade",
    }
    return mapping.get(str(family), str(family))


def _combo_key(*, symbol: str, timeframe: str, strategy: str, exit_type: str, cost_level: str) -> str:
    return stable_hash(
        {
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "strategy": str(strategy),
            "exit_type": str(exit_type),
            "cost_level": str(cost_level),
        },
        length=20,
    )


def _write_docs_report(leaderboard: pd.DataFrame, summary: dict[str, Any], report_md: Path, report_json: Path) -> None:
    validate_stage12_summary_schema(summary)
    report_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-12 Full Price-Family Robustness Sweep")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- symbols: `{', '.join(summary['symbols'])}`")
    lines.append(f"- timeframes: `{', '.join(summary['timeframes'])}`")
    lines.append(f"- total_combinations: `{summary['total_combinations']}`")
    lines.append(f"- valid_combinations: `{summary['valid_combinations']}`")
    lines.append(f"- runtime_seconds: `{float(summary['runtime_seconds']):.3f}`")
    lines.append(f"- verdict: `{summary['verdict']}`")
    lines.append("")
    lines.append("## Runtime Breakdown by Timeframe")
    lines.append("| timeframe | runtime_seconds |")
    lines.append("| --- | ---: |")
    for timeframe, seconds in sorted(summary.get("runtime_by_timeframe", {}).items()):
        lines.append(f"| {timeframe} | {float(seconds):.4f} |")
    lines.append("")

    lines.append("## Top 10 Robust Combinations")
    lines.append("| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb | PF | expectancy | robust_score | cost_sensitivity | stability | MC_p_ruin | MC_p_return_negative |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
    for row in summary.get("top_robust", []):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {row.get('strategy','')} | "
            f"{row.get('exit_type','')} | {row.get('cost_level','')} | {_safe_float(row.get('exp_lcb',0.0)):.6f} | "
            f"{_safe_float(row.get('PF',0.0)):.6f} | {_safe_float(row.get('expectancy',0.0)):.6f} | "
            f"{_safe_float(row.get('robust_score',0.0)):.6f} | {_safe_float(row.get('cost_sensitivity_slope',0.0)):.6f} | "
            f"{row.get('stability_classification','')} | {_safe_float(row.get('MC_p_ruin', math.nan)):.6f} | "
            f"{_safe_float(row.get('MC_p_return_negative', math.nan)):.6f} |"
        )
    lines.append("")

    lines.append("## Per-Timeframe Summary")
    tf_summary = (
        leaderboard.groupby("timeframe", dropna=False)
        .agg(
            combos=("combo_key", "count"),
            valid=("is_valid", "sum"),
            exp_lcb_median=("exp_lcb", "median"),
            robust_top=("robust_score", "max"),
        )
        .reset_index()
        .sort_values("timeframe")
    )
    lines.append("| timeframe | combos | valid | exp_lcb_median | best_robust_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in tf_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['timeframe']} | {int(row['combos'])} | {int(row['valid'])} | "
            f"{float(row['exp_lcb_median']):.6f} | {float(row['robust_top']):.6f} |"
        )
    lines.append("")

    lines.append("## Per-Symbol Summary")
    sym_summary = (
        leaderboard.groupby("symbol", dropna=False)
        .agg(
            combos=("combo_key", "count"),
            valid=("is_valid", "sum"),
            exp_lcb_median=("exp_lcb", "median"),
            robust_top=("robust_score", "max"),
        )
        .reset_index()
        .sort_values("symbol")
    )
    lines.append("| symbol | combos | valid | exp_lcb_median | best_robust_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in sym_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['symbol']} | {int(row['combos'])} | {int(row['valid'])} | "
            f"{float(row['exp_lcb_median']):.6f} | {float(row['robust_top']):.6f} |"
        )
    lines.append("")

    lines.append("## Stability Heatmap Table")
    heat = (
        leaderboard.pivot_table(
            index=["symbol", "timeframe"],
            columns="stability_classification",
            values="combo_key",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .sort_values(["symbol", "timeframe"])
    )
    lines.append("| symbol | timeframe | STABLE | UNSTABLE | INVALID | INSUFFICIENT_DATA |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in heat.to_dict(orient="records"):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {int(row.get('STABLE',0))} | "
            f"{int(row.get('UNSTABLE',0))} | {int(row.get('INVALID',0))} | {int(row.get('INSUFFICIENT_DATA',0))} |"
        )
    lines.append("")

    lines.append("## Cost Sensitivity Chart Data")
    lines.append("| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb |")
    lines.append("| --- | --- | --- | --- | --- | ---: |")
    plot_rows = leaderboard[
        ["symbol", "timeframe", "strategy", "exit_type", "cost_level", "exp_lcb"]
    ].sort_values(["symbol", "timeframe", "strategy", "exit_type", "cost_level"])
    for row in plot_rows.to_dict(orient="records"):
        lines.append(
            f"| {row['symbol']} | {row['timeframe']} | {row['strategy']} | {row['exit_type']} | "
            f"{row['cost_level']} | {float(row['exp_lcb']):.6f} |"
        )
    lines.append("")

    mc_rows = leaderboard[leaderboard["MC_p_ruin"].notna()].copy()
    lines.append("## Monte Carlo Validation Summary (Top 20%)")
    if mc_rows.empty:
        lines.append("- no combinations selected for Monte Carlo")
    else:
        lines.append("| symbol | timeframe | strategy | exit_type | cost_level | MC_p_ruin | MC_p_return_negative | MC_maxDD_p95 | MC_expected_log_growth |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in mc_rows.sort_values("robust_score", ascending=False).to_dict(orient="records"):
            lines.append(
                f"| {row['symbol']} | {row['timeframe']} | {row['strategy']} | {row['exit_type']} | {row['cost_level']} | "
                f"{float(row['MC_p_ruin']):.6f} | {float(row['MC_p_return_negative']):.6f} | "
                f"{float(row['MC_maxDD_p95']):.6f} | {float(row['MC_expected_log_growth']):.6f} |"
            )
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- `{summary['verdict']}`")

    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _finite(value: Any, default: float = 0.0, clip: float | None = None) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if not np.isfinite(number):
        number = float(default)
    if clip is not None and number > float(clip):
        number = float(clip)
    return float(number)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)
