"""Stage-11.5 structural benchmark with deterministic cache diagnostics."""

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
from buffmini.data.cache import FeatureFrameCache, compute_features_cached, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.data.storage import save_parquet
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-11.5 deterministic engine benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--tfs", type=str, default="15m,1h,2h,4h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-rows", type=int, default=6000)
    return parser.parse_args()


def run_stage11_5_bench(
    *,
    config: dict[str, Any],
    symbols: list[str],
    base_timeframe: str,
    tfs: list[str],
    seed: int,
    data_dir: Path,
    derived_dir: Path,
    runs_dir: Path,
    dry_run: bool,
    dry_run_rows: int,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["search"]["seed"] = int(seed)
    cfg["universe"]["symbols"] = list(symbols)
    cfg["universe"]["base_timeframe"] = str(base_timeframe)
    cfg["universe"]["timeframe"] = str(tfs[0] if tfs else "1h")
    cfg["universe"]["operational_timeframe"] = str(tfs[0] if tfs else "1h")
    cfg["data"]["resample_source"] = "base" if str(base_timeframe) != str(tfs[0] if tfs else "1h") else "direct"

    if bool(dry_run):
        _seed_synthetic_data(
            symbols=symbols,
            timeframe=str(base_timeframe),
            rows=int(dry_run_rows),
            seed=int(seed),
            data_dir=data_dir,
        )

    first = _bench_once(cfg=cfg, symbols=symbols, tfs=tfs, data_dir=data_dir, derived_dir=derived_dir)
    second = _bench_once(cfg=cfg, symbols=symbols, tfs=tfs, data_dir=data_dir, derived_dir=derived_dir)

    payload = {
        "seed": int(seed),
        "base_timeframe": str(base_timeframe),
        "tfs": list(tfs),
        "symbols": list(symbols),
        "config_hash": compute_config_hash(cfg),
        "first_run_seconds": float(first["total_seconds"]),
        "second_run_seconds": float(second["total_seconds"]),
        "speedup_factor": (
            float(first["total_seconds"] / second["total_seconds"])
            if float(second["total_seconds"]) > 0
            else 0.0
        ),
        "derived_cache_hit_rate": float(second["cache"]["derived_hit_rate"]),
        "feature_cache_hit_rate": float(second["cache"]["feature_hit_rate"]),
        "first_breakdown": dict(first["breakdown"]),
        "second_breakdown": dict(second["breakdown"]),
        "first_cache": dict(first["cache"]),
        "second_cache": dict(second["cache"]),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage11_5_bench"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload["run_id"] = run_id
    payload["config_hash"] = compute_config_hash(cfg)
    payload["data_hash"] = _aggregate_data_hash(first["data_hash_by_tf"])
    payload["data_hash_by_tf"] = dict(first["data_hash_by_tf"])
    (run_dir / "perf_meta.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def _bench_once(
    *,
    cfg: dict[str, Any],
    symbols: list[str],
    tfs: list[str],
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    breakdown = {"load": 0.0, "features": 0.0, "backtest": 0.0}
    data_hash_by_tf: dict[str, str] = {}
    feature_cache = FeatureFrameCache()
    strategy = trend_pullback()
    resolved_end_ts = str(cfg.get("universe", {}).get("resolved_end_ts") or "")
    config_hash = compute_config_hash(cfg)

    # Store is built once and reused for all symbols/timeframes.
    t0 = time.perf_counter()
    store = build_data_store(
        backend=str(cfg.get("data", {}).get("backend", "parquet")),
        data_dir=data_dir,
        base_timeframe=str(cfg.get("universe", {}).get("base_timeframe") or "1m"),
        resample_source="base",
        derived_dir=derived_dir,
        partial_last_bucket=bool(cfg.get("data", {}).get("partial_last_bucket", False)),
        config_hash=config_hash,
        resolved_end_ts=resolved_end_ts,
    )
    breakdown["load"] += float(time.perf_counter() - t0)

    for timeframe in tfs:
        for symbol in symbols:
            t_load = time.perf_counter()
            raw = store.load_ohlcv(symbol=symbol, timeframe=str(timeframe))
            breakdown["load"] += float(time.perf_counter() - t_load)
            if raw.empty:
                continue
            data_hash = ohlcv_data_hash(raw)
            data_hash_by_tf[f"{symbol}|{timeframe}"] = data_hash

            feature_config_hash = stable_hash(
                {
                    "timeframe": str(timeframe),
                    "include_futures_extras": bool(cfg.get("data", {}).get("include_futures_extras", False)),
                    "futures_extras": cfg.get("data", {}).get("futures_extras", {}),
                    "cost_model": cfg.get("cost_model", {}),
                },
                length=16,
            )
            t_feat = time.perf_counter()
            features, _, _ = compute_features_cached(
                cache=feature_cache,
                symbol=str(symbol),
                timeframe=str(timeframe),
                resolved_end_ts=resolved_end_ts,
                feature_config_hash=feature_config_hash,
                data_hash=data_hash,
                builder=lambda f=raw, s=symbol, tf=timeframe: calculate_features(
                    f,
                    config=cfg,
                    symbol=s,
                    timeframe=str(tf),
                    derived_data_dir=derived_dir,
                ),
            )
            breakdown["features"] += float(time.perf_counter() - t_feat)

            t_bt = time.perf_counter()
            features = features.copy()
            features["signal"] = generate_signals(features, strategy=strategy, gating_mode="none")
            run_backtest(
                frame=features,
                strategy_name=strategy.name,
                symbol=str(symbol),
                stop_atr_multiple=1.5,
                take_profit_atr_multiple=3.0,
                max_hold_bars=24,
                round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]),
                slippage_pct=float(cfg["costs"]["slippage_pct"]),
                initial_capital=10000.0,
                cost_model_cfg=cfg.get("cost_model"),
                engine_mode="numpy",
            )
            breakdown["backtest"] += float(time.perf_counter() - t_bt)

    total_seconds = float(time.perf_counter() - started_total)
    derived_stats = getattr(store, "derived_cache", None).stats if hasattr(store, "derived_cache") else None
    derived_hits = int(getattr(derived_stats, "hits", 0))
    derived_misses = int(getattr(derived_stats, "misses", 0))
    derived_total = derived_hits + derived_misses
    feature_hits = int(feature_cache.stats.hits)
    feature_misses = int(feature_cache.stats.misses)
    feature_total = feature_hits + feature_misses

    return {
        "total_seconds": total_seconds,
        "breakdown": breakdown,
        "cache": {
            "derived_hits": derived_hits,
            "derived_misses": derived_misses,
            "derived_hit_rate": 0.0 if derived_total <= 0 else float(derived_hits / derived_total),
            "feature_hits": feature_hits,
            "feature_misses": feature_misses,
            "feature_hit_rate": 0.0 if feature_total <= 0 else float(feature_hits / feature_total),
            "features_compute_calls_per_tf": dict(feature_cache.stats.compute_calls_per_tf),
        },
        "data_hash_by_tf": data_hash_by_tf,
    }


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


def _aggregate_data_hash(data_hash_by_tf: dict[str, str]) -> str:
    return stable_hash(dict(sorted(data_hash_by_tf.items())), length=16)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    tfs = [item.strip().lower() for item in str(args.tfs).split(",") if item.strip()]
    payload = run_stage11_5_bench(
        config=config,
        symbols=symbols,
        base_timeframe=str(args.base_timeframe),
        tfs=tfs,
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        runs_dir=args.runs_dir,
        dry_run=bool(args.dry_run),
        dry_run_rows=int(args.dry_run_rows),
    )
    print(f"run_id: {payload['run_id']}")
    print(f"first_run_seconds: {float(payload['first_run_seconds']):.4f}")
    print(f"second_run_seconds: {float(payload['second_run_seconds']):.4f}")
    print(f"derived_cache_hit_rate: {float(payload['derived_cache_hit_rate']):.4f}")
    print(f"feature_cache_hit_rate: {float(payload['feature_cache_hit_rate']):.4f}")


if __name__ == "__main__":
    main()
