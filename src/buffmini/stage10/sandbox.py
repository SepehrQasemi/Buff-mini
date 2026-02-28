"""Stage-10.6 sandbox ranking for entry signal families."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import compute_config_hash, get_universe_end
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.cache import FeatureFrameCache, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.stage10.signals import SIGNAL_FAMILIES, generate_signal_family, signal_family_type
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.leakage_harness import synthetic_ohlcv


def run_stage10_sandbox(
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    cost_mode: str = "v2",
    exit_mode: str = "fixed_atr",
    top_k_per_category: int = 2,
    runs_root: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Evaluate Stage-10 signal families in isolation and rank robustly."""

    cfg = _normalize_sandbox_config(config=config, cost_mode=cost_mode, exit_mode=exit_mode)
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    if str(timeframe) != "1h":
        raise ValueError("Stage-10 sandbox currently supports timeframe=1h only")

    features_by_symbol = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe="1h",
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    if not features_by_symbol:
        raise ValueError("No features available for sandbox ranking")

    rows: list[dict[str, Any]] = []
    bootstrap_resamples = int(cfg["evaluation"]["stage10"]["sandbox"].get("bootstrap_resamples", 500))
    for family in SIGNAL_FAMILIES:
        family_rows: list[dict[str, Any]] = []
        for symbol, frame in features_by_symbol.items():
            metrics = _evaluate_family_symbol(
                frame=frame,
                symbol=symbol,
                family=family,
                cfg=cfg,
                seed=int(seed),
                bootstrap_resamples=bootstrap_resamples,
            )
            family_rows.append(metrics)
            row = {"family": family, "symbol": symbol, "category": signal_family_type(family)}
            row.update(metrics)
            rows.append(row)

        if family_rows:
            rows.append(_aggregate_family_row(family=family, family_rows=family_rows))

    rankings = pd.DataFrame(rows)
    family_rankings = rankings.loc[rankings["symbol"] == "ALL"].copy().reset_index(drop=True)
    family_rankings = family_rankings.sort_values(
        by=["score", "exp_lcb_proxy", "family"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    family_rankings["rank"] = np.arange(1, len(family_rankings) + 1, dtype=int)

    enabled, disabled = _select_enabled_signals(family_rankings=family_rankings, top_k_per_category=int(top_k_per_category))
    data_hash = _compute_data_hash(features_by_symbol)
    config_hash = compute_config_hash(cfg)
    resolved_end_ts = _resolve_end_ts(cfg, features_by_symbol)

    payload = {
        "symbols": resolved_symbols,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "cost_mode": str(cost_mode),
        "exit_mode": str(exit_mode),
        "top_k_per_category": int(top_k_per_category),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "resolved_end_ts": resolved_end_ts,
        "enabled_signals": enabled,
        "disabled_signals": disabled,
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage10_sandbox"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(run_dir / "sandbox_rankings.csv", index=False)
    per_signal_metrics = _build_per_signal_metrics(rankings=rankings)
    (run_dir / "per_signal_metrics.json").write_text(
        json.dumps(per_signal_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    summary = {
        "stage": "10.6_sandbox",
        "run_id": run_id,
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "cost_mode": str(cost_mode),
        "exit_mode": str(exit_mode),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "top_k_per_category": int(top_k_per_category),
        "enabled_signals": enabled,
        "disabled_signals": disabled,
        "rank_table_path": str(run_dir / "sandbox_rankings.csv"),
    }
    (run_dir / "sandbox_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def _evaluate_family_symbol(
    frame: pd.DataFrame,
    symbol: str,
    family: str,
    cfg: dict[str, Any],
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, float]:
    signal_frame = generate_signal_family(frame=frame, family=family, params=cfg["evaluation"]["stage10"]["signals"]["defaults"].get(family, {}))
    work = frame.copy()
    work["signal"] = signal_frame["signal"].astype(int)

    base_result = run_backtest(
        frame=work,
        strategy_name=f"Sandbox::{family}",
        symbol=symbol,
        exit_mode=str(cfg["evaluation"]["stage10"]["sandbox"]["exit_mode"]),
        max_hold_bars=int(cfg["evaluation"]["stage10"]["evaluation"]["max_hold_bars"]),
        stop_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["stop_atr_multiple"]),
        take_profit_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["take_profit_atr_multiple"]),
        round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
        slippage_pct=float(cfg["costs"]["slippage_pct"]),
        cost_model_cfg=cfg["cost_model"],
    )
    stress_cfg = _stress_cost_cfg(cfg["cost_model"])
    stress_result = run_backtest(
        frame=work,
        strategy_name=f"SandboxStress::{family}",
        symbol=symbol,
        exit_mode=str(cfg["evaluation"]["stage10"]["sandbox"]["exit_mode"]),
        max_hold_bars=int(cfg["evaluation"]["stage10"]["evaluation"]["max_hold_bars"]),
        stop_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["stop_atr_multiple"]),
        take_profit_atr_multiple=float(cfg["evaluation"]["stage10"]["evaluation"]["take_profit_atr_multiple"]),
        round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
        slippage_pct=float(cfg["costs"]["slippage_pct"]),
        cost_model_cfg=stress_cfg,
    )

    base = dict(base_result.metrics)
    stress = dict(stress_result.metrics)
    trade_count = float(base.get("trade_count", 0.0))
    expectancy = float(base.get("expectancy", 0.0))
    profit_factor = _finite(base.get("profit_factor", 0.0), default=0.0, clip=10.0)
    max_drawdown = _finite(base.get("max_drawdown", 0.0), default=0.0)
    trades_per_month = _trades_per_month(base_result.trades)
    exp_lcb_proxy = _bootstrap_exp_lcb(
        trades=base_result.trades,
        resamples=int(bootstrap_resamples),
        seed=int.from_bytes(
            stable_hash(f"{seed}:{symbol}:{family}:exp_lcb", length=12).encode("utf-8"),
            "little",
            signed=False,
        )
        % (2**31),
    )
    drag_penalty = _drag_penalty(base=base, stress=stress)
    score = exp_lcb_proxy - (0.5 * drag_penalty) - (0.2 * max_drawdown)

    return {
        "trade_count": trade_count,
        "trades_per_month": trades_per_month,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "exp_lcb_proxy": exp_lcb_proxy,
        "drag_penalty": drag_penalty,
        "score": float(score),
        "stress_expectancy": _finite(stress.get("expectancy", 0.0), default=0.0),
        "stress_profit_factor": _finite(stress.get("profit_factor", 0.0), default=0.0, clip=10.0),
    }


def _aggregate_family_row(family: str, family_rows: list[dict[str, float]]) -> dict[str, Any]:
    frame = pd.DataFrame(family_rows)
    weights = frame["trade_count"].clip(lower=0.0).to_numpy(dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(frame), dtype=float)
    weights = weights / float(weights.sum())
    row = {
        "family": family,
        "symbol": "ALL",
        "category": signal_family_type(family),
        "trade_count": float(frame["trade_count"].sum()),
        "trades_per_month": float(np.sum(weights * frame["trades_per_month"].to_numpy(dtype=float))),
        "profit_factor": float(np.sum(weights * frame["profit_factor"].to_numpy(dtype=float))),
        "expectancy": float(np.sum(weights * frame["expectancy"].to_numpy(dtype=float))),
        "max_drawdown": float(np.sum(weights * frame["max_drawdown"].to_numpy(dtype=float))),
        "exp_lcb_proxy": float(np.sum(weights * frame["exp_lcb_proxy"].to_numpy(dtype=float))),
        "drag_penalty": float(np.sum(weights * frame["drag_penalty"].to_numpy(dtype=float))),
        "score": float(np.sum(weights * frame["score"].to_numpy(dtype=float))),
        "stress_expectancy": float(np.sum(weights * frame["stress_expectancy"].to_numpy(dtype=float))),
        "stress_profit_factor": float(np.sum(weights * frame["stress_profit_factor"].to_numpy(dtype=float))),
    }
    return row


def _select_enabled_signals(family_rankings: pd.DataFrame, top_k_per_category: int) -> tuple[list[str], list[str]]:
    enabled: list[str] = []
    for category in ("trend", "mean_reversion", "breakout"):
        subset = family_rankings.loc[family_rankings["category"] == category].head(int(top_k_per_category))
        enabled.extend(subset["family"].astype(str).tolist())
    enabled_unique = sorted(dict.fromkeys(enabled))
    disabled = sorted([family for family in SIGNAL_FAMILIES if family not in enabled_unique])
    return enabled_unique, disabled


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
    feature_cache_enabled = bool(config.get("data", {}).get("feature_cache", {}).get("enabled", True))
    feature_cache = FeatureFrameCache() if feature_cache_enabled else None
    rows = int(config["evaluation"]["stage10"]["evaluation"].get("dry_run_rows", 2400))
    if dry_run:
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


def _stress_cost_cfg(cost_model_cfg: dict[str, Any]) -> dict[str, Any]:
    stressed = json.loads(json.dumps(cost_model_cfg))
    stressed.setdefault("mode", "v2")
    stressed.setdefault("v2", {})
    v2 = stressed["v2"]
    v2["delay_bars"] = int(v2.get("delay_bars", 0)) + 1
    v2["spread_bps"] = float(v2.get("spread_bps", 0.0)) + 1.0
    v2["slippage_bps_base"] = float(v2.get("slippage_bps_base", 0.0)) + 1.0
    return stressed


def _drag_penalty(base: dict[str, Any], stress: dict[str, Any]) -> float:
    base_expectancy = _finite(base.get("expectancy", 0.0), default=0.0)
    stress_expectancy = _finite(stress.get("expectancy", 0.0), default=0.0)
    return float(max(0.0, base_expectancy - stress_expectancy))


def _expectancy_lcb(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna().to_numpy(dtype=float)
    if pnl.size == 0:
        return 0.0
    mean = float(np.mean(pnl))
    if pnl.size == 1:
        return mean
    std = float(np.std(pnl, ddof=0))
    return float(mean - (std / math.sqrt(float(pnl.size))))


def _bootstrap_exp_lcb(trades: pd.DataFrame, resamples: int, seed: int) -> float:
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna().to_numpy(dtype=float)
    if pnl.size == 0:
        return 0.0
    if pnl.size == 1:
        return float(pnl[0])
    rng = np.random.default_rng(int(seed))
    sample_size = int(pnl.size)
    means = np.empty(int(resamples), dtype=float)
    for idx in range(int(resamples)):
        picks = rng.integers(0, sample_size, size=sample_size)
        means[idx] = float(np.mean(pnl[picks]))
    return float(np.quantile(means, 0.05))


def _trades_per_month(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    ts = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span_days = max(1.0, float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0))
    return float(len(ts) / (span_days / 30.0))


def _build_per_signal_metrics(rankings: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in SIGNAL_FAMILIES:
        subset = rankings.loc[(rankings["family"] == family) & (rankings["symbol"] == "ALL")]
        if subset.empty:
            continue
        output[family] = {key: _safe_json_value(val) for key, val in subset.iloc[0].to_dict().items()}
    return output


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


def _normalize_sandbox_config(config: dict[str, Any], cost_mode: str, exit_mode: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("evaluation", {}).setdefault("stage10", {})
    stage10 = cfg["evaluation"]["stage10"]
    stage10.setdefault("sandbox", {})
    stage10["sandbox"]["exit_mode"] = str(normalize_exit_mode(str(exit_mode)))
    stage10["cost_mode"] = str(cost_mode)
    cfg.setdefault("cost_model", {})
    cfg["cost_model"]["mode"] = str(cost_mode)
    return cfg


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(numeric):
        return 0.0
    return numeric


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
