"""Benchmark core engine data->features->backtest path with cache telemetry."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.cache import FeatureComputeSession, FeatureFrameCache, cache_limits_from_config, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.data.storage import save_parquet
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Buff-mini engine stages with caching")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--operational-timeframe", type=str, default="1h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Generate deterministic synthetic bars in data-dir first.")
    parser.add_argument("--dry-run-rows", type=int, default=6000, help="Synthetic 1m bars per symbol.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    cfg = deepcopy(config)
    cfg["universe"]["symbols"] = symbols
    cfg["universe"]["base_timeframe"] = str(args.base_timeframe)
    cfg["universe"]["operational_timeframe"] = str(args.operational_timeframe)
    cfg["universe"]["timeframe"] = str(args.operational_timeframe)
    cfg["data"]["resample_source"] = "base" if str(args.base_timeframe) != str(args.operational_timeframe) else "direct"
    cfg["search"]["seed"] = int(args.seed)

    if bool(args.dry_run):
        _seed_synthetic_data(
            symbols=symbols,
            timeframe=str(args.base_timeframe),
            rows=int(args.dry_run_rows),
            seed=int(args.seed),
            data_dir=args.data_dir,
        )

    first = _bench_once(cfg=cfg, data_dir=args.data_dir, derived_dir=args.derived_dir)
    second = _bench_once(cfg=cfg, data_dir=args.data_dir, derived_dir=args.derived_dir)

    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'base': args.base_timeframe, 'op': args.operational_timeframe}, length=12)}_stage11_4_bench"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "seed": int(args.seed),
        "config_hash": compute_config_hash(cfg),
        "data_hash": _data_hash(first["frames"]),
        "base_timeframe": str(args.base_timeframe),
        "operational_timeframe": str(args.operational_timeframe),
        "first_run": first["profile"],
        "second_run": second["profile"],
        "cache_hit_rate_rerun": float(second["profile"]["cache"]["combined_hit_rate"]),
        "metrics": second["metrics"],
    }
    (run_dir / "perf_profile.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"first_total_seconds: {first['profile']['total_seconds']:.4f}")
    print(f"second_total_seconds: {second['profile']['total_seconds']:.4f}")
    print(f"cache_hit_rate_rerun: {summary['cache_hit_rate_rerun']:.4f}")
    print(f"perf_profile: {run_dir / 'perf_profile.json'}")


def _bench_once(cfg: dict[str, Any], data_dir: Path, derived_dir: Path) -> dict[str, Any]:
    timings: dict[str, float] = {}
    start_total = time.perf_counter()

    t0 = time.perf_counter()
    cache_limits = cache_limits_from_config(cfg)
    store = build_data_store(
        backend=str(cfg.get("data", {}).get("backend", "parquet")),
        data_dir=data_dir,
        base_timeframe=str(cfg["universe"]["base_timeframe"]),
        resample_source=str(cfg.get("data", {}).get("resample_source", "direct")),
        derived_dir=derived_dir,
        partial_last_bucket=bool(cfg.get("data", {}).get("partial_last_bucket", False)),
        config_hash=compute_config_hash(cfg),
        resolved_end_ts=str(cfg.get("universe", {}).get("resolved_end_ts") or ""),
        cache_limits=cache_limits,
    )
    symbols = list(cfg["universe"]["symbols"])
    operational_tf = str(cfg["universe"]["operational_timeframe"])
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames[symbol] = store.load_ohlcv(symbol=symbol, timeframe=operational_tf)
    timings["load_seconds"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    feature_cache = FeatureFrameCache(limits=cache_limits)
    feature_session = FeatureComputeSession(feature_cache)
    features_by_symbol: dict[str, pd.DataFrame] = {}
    resolved_end_ts = str(cfg.get("universe", {}).get("resolved_end_ts") or "")
    for symbol, frame in frames.items():
        data_hash = ohlcv_data_hash(frame)
        feature_config_hash = stable_hash(
            {
                "timeframe": operational_tf,
                "include_futures_extras": bool(cfg.get("data", {}).get("include_futures_extras", False)),
                "futures_extras": cfg.get("data", {}).get("futures_extras", {}),
                "cost_model": cfg.get("cost_model", {}),
            },
            length=16,
        )
        features, _, _ = feature_session.get_or_build(
            symbol=str(symbol),
            timeframe=str(operational_tf),
            resolved_end_ts=resolved_end_ts,
            feature_config_hash=str(feature_config_hash),
            data_hash=str(data_hash),
            builder=lambda r=frame, s=symbol: calculate_features(
                r,
                config=cfg,
                symbol=s,
                timeframe=operational_tf,
                derived_data_dir=derived_dir,
            ),
        )
        features_by_symbol[symbol] = features
    timings["features_seconds"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    strategy = trend_pullback()
    metrics_rows: list[dict[str, float]] = []
    for symbol, frame in features_by_symbol.items():
        work = frame.copy()
        work["signal"] = generate_signals(work, strategy=strategy, gating_mode="none")
        result = run_backtest(
            frame=work,
            strategy_name=strategy.name,
            symbol=symbol,
            stop_atr_multiple=1.5,
            take_profit_atr_multiple=3.0,
            max_hold_bars=24,
            round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
            slippage_pct=float(cfg["costs"]["slippage_pct"]),
            initial_capital=10000.0,
            cost_model_cfg=cfg.get("cost_model"),
        )
        metrics_rows.append(
            {
                "trade_count": float(result.metrics.get("trade_count", 0.0)),
                "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
                "expectancy": float(result.metrics.get("expectancy", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            }
        )
    timings["backtest_seconds"] = time.perf_counter() - t2
    timings["reports_seconds"] = 0.0
    timings["total_seconds"] = time.perf_counter() - start_total

    aggregate = _aggregate_metrics(metrics_rows)
    cache_stats = {
        "derived_hits": int(getattr(store, "derived_cache", None).stats.hits if hasattr(store, "derived_cache") else 0),
        "derived_misses": int(getattr(store, "derived_cache", None).stats.misses if hasattr(store, "derived_cache") else 0),
        "feature_hits": int(feature_cache.stats.hits),
        "feature_misses": int(feature_cache.stats.misses),
    }
    total_cache_ops = cache_stats["derived_hits"] + cache_stats["derived_misses"] + cache_stats["feature_hits"] + cache_stats["feature_misses"]
    combined_hit_rate = 0.0 if total_cache_ops <= 0 else float((cache_stats["derived_hits"] + cache_stats["feature_hits"]) / total_cache_ops)
    cache_stats["combined_hit_rate"] = combined_hit_rate

    profile = {
        "timings": timings,
        "load_seconds": timings["load_seconds"],
        "resample_seconds": 0.0,
        "features_seconds": timings["features_seconds"],
        "joins_seconds": 0.0,
        "backtest_seconds": timings["backtest_seconds"],
        "reports_seconds": timings["reports_seconds"],
        "total_seconds": timings["total_seconds"],
        "cache": cache_stats,
    }
    return {"profile": profile, "metrics": aggregate, "frames": frames}


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"trade_count": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "max_drawdown": 0.0}
    frame = pd.DataFrame(rows)
    return {
        "trade_count": float(frame["trade_count"].sum()),
        "profit_factor": float(frame["profit_factor"].mean()),
        "expectancy": float(frame["expectancy"].mean()),
        "max_drawdown": float(frame["max_drawdown"].mean()),
    }


def _data_hash(frames: dict[str, pd.DataFrame]) -> str:
    payload = {
        symbol: {
            "rows": int(len(frame)),
            "hash": ohlcv_data_hash(frame),
        }
        for symbol, frame in sorted(frames.items())
    }
    return stable_hash(payload, length=16)


def _seed_synthetic_data(
    *,
    symbols: list[str],
    timeframe: str,
    rows: int,
    seed: int,
    data_dir: Path,
) -> None:
    rows = max(int(rows), 300)
    base_start = pd.Timestamp("2025-01-01T00:00:00Z")
    delta = _timeframe_delta(timeframe)
    for idx, symbol in enumerate(symbols):
        rng = np.random.default_rng(int(seed) + idx * 1009)
        timestamps = pd.date_range(start=base_start, periods=rows, freq=delta, tz="UTC")
        log_returns = rng.normal(loc=0.0, scale=0.0015, size=rows)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        spread = rng.uniform(0.0002, 0.0025, size=rows)
        high = np.maximum(open_, close) * (1.0 + spread)
        low = np.minimum(open_, close) * (1.0 - spread)
        volume = rng.uniform(50.0, 2000.0, size=rows)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        save_parquet(frame=frame, symbol=symbol, timeframe=timeframe, data_dir=data_dir)


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip().lower()
    if text.endswith("m"):
        return pd.Timedelta(minutes=int(text[:-1]))
    if text.endswith("h"):
        return pd.Timedelta(hours=int(text[:-1]))
    if text.endswith("d"):
        return pd.Timedelta(days=int(text[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


if __name__ == "__main__":
    main()
