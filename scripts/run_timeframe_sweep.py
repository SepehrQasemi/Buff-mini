"""Run deterministic operational-timeframe sweep from a 1m base dataset."""

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
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.cache import FeatureFrameCache, compute_features_cached, ohlcv_data_hash
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run timeframe sweep derived from 1m base data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--tfs", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--out-md", type=Path, default=Path("docs") / "timeframe_sweep_from_1m.md")
    parser.add_argument("--out-json", type=Path, default=Path("docs") / "timeframe_sweep_from_1m.json")
    return parser.parse_args()


def run_timeframe_sweep(
    *,
    config: dict[str, Any],
    symbols: list[str],
    base_timeframe: str,
    operational_timeframes: list[str],
    seed: int,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for timeframe in operational_timeframes:
        cfg = deepcopy(config)
        cfg["search"]["seed"] = int(seed)
        cfg["universe"]["symbols"] = list(symbols)
        cfg["universe"]["base_timeframe"] = str(base_timeframe)
        cfg["universe"]["operational_timeframe"] = str(timeframe)
        cfg["universe"]["timeframe"] = str(timeframe)
        cfg["data"]["resample_source"] = "base" if str(base_timeframe) != str(timeframe) else "direct"

        started = time.perf_counter()
        metrics = _evaluate_single_timeframe(cfg=cfg, symbols=symbols, timeframe=str(timeframe), data_dir=data_dir, derived_dir=derived_dir)
        runtime = float(time.perf_counter() - started)
        row = {
            "timeframe": str(timeframe),
            "trade_count": float(metrics["trade_count"]),
            "trades_per_month": float(metrics["trades_per_month"]),
            "profit_factor": float(metrics["profit_factor"]),
            "expectancy": float(metrics["expectancy"]),
            "exp_lcb": float(metrics["exp_lcb"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "runtime_seconds": runtime,
            "data_hash": str(metrics["data_hash"]),
        }
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("timeframe").reset_index(drop=True)
    best_pf = frame.loc[frame["profit_factor"].idxmax()].to_dict() if not frame.empty else None
    best_exp_lcb = frame.loc[frame["exp_lcb"].idxmax()].to_dict() if not frame.empty else None
    return {
        "seed": int(seed),
        "base_timeframe": str(base_timeframe),
        "symbols": list(symbols),
        "rows": frame.to_dict(orient="records"),
        "best_by_profit_factor": best_pf,
        "best_by_exp_lcb": best_exp_lcb,
        "config_hash": compute_config_hash(config),
    }


def _evaluate_single_timeframe(
    *,
    cfg: dict[str, Any],
    symbols: list[str],
    timeframe: str,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, float]:
    feature_cache_enabled = bool(cfg.get("data", {}).get("feature_cache", {}).get("enabled", True))
    feature_cache = FeatureFrameCache() if feature_cache_enabled else None
    store = build_data_store(
        backend=str(cfg.get("data", {}).get("backend", "parquet")),
        data_dir=data_dir,
        base_timeframe=str(cfg["universe"]["base_timeframe"]),
        resample_source=str(cfg["data"]["resample_source"]),
        derived_dir=derived_dir,
        partial_last_bucket=bool(cfg.get("data", {}).get("partial_last_bucket", False)),
        config_hash=compute_config_hash(cfg),
        resolved_end_ts=str(cfg.get("universe", {}).get("resolved_end_ts") or ""),
    )
    strategy = trend_pullback()
    metrics_rows: list[dict[str, float]] = []
    pnls: list[float] = []
    hashes: list[str] = []
    resolved_end_ts = str(cfg.get("universe", {}).get("resolved_end_ts") or "")
    for symbol in symbols:
        frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        hashes.append(ohlcv_data_hash(frame))
        feature_config_hash = stable_hash(
            {
                "timeframe": str(timeframe),
                "include_futures_extras": bool(cfg.get("data", {}).get("include_futures_extras", False)),
                "futures_extras": cfg.get("data", {}).get("futures_extras", {}),
                "cost_model": cfg.get("cost_model", {}),
            },
            length=16,
        )
        features, _, _ = compute_features_cached(
            cache=feature_cache,
            symbol=str(symbol),
            timeframe=str(timeframe),
            resolved_end_ts=resolved_end_ts,
            feature_config_hash=str(feature_config_hash),
            data_hash=str(hashes[-1]),
            builder=lambda f=frame, s=symbol: calculate_features(
                f,
                config=cfg,
                symbol=s,
                timeframe=timeframe,
                derived_data_dir=derived_dir,
            ),
        )
        features["signal"] = generate_signals(features, strategy=strategy, gating_mode="none")
        result = run_backtest(
            frame=features,
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
        trade_count = float(result.metrics.get("trade_count", 0.0))
        window_months = _estimate_months(features)
        metrics_rows.append(
            {
                "trade_count": trade_count,
                "trades_per_month": trade_count / max(window_months, 1e-6),
                "profit_factor": float(result.metrics.get("profit_factor", 0.0)),
                "expectancy": float(result.metrics.get("expectancy", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            }
        )
        if not result.trades.empty and "pnl" in result.trades.columns:
            pnls.extend(pd.to_numeric(result.trades["pnl"], errors="coerce").dropna().tolist())

    if not metrics_rows:
        return {
            "trade_count": 0.0,
            "trades_per_month": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "exp_lcb": 0.0,
            "max_drawdown": 0.0,
            "data_hash": stable_hash({"hashes": hashes}, length=16),
        }

    frame = pd.DataFrame(metrics_rows)
    return {
        "trade_count": float(frame["trade_count"].sum()),
        "trades_per_month": float(frame["trades_per_month"].mean()),
        "profit_factor": float(frame["profit_factor"].mean()),
        "expectancy": float(frame["expectancy"].mean()),
        "exp_lcb": _exp_lcb(pnls),
        "max_drawdown": float(frame["max_drawdown"].mean()),
        "data_hash": stable_hash({"hashes": hashes}, length=16),
    }


def _estimate_months(features: pd.DataFrame) -> float:
    if features.empty:
        return 0.0
    ts = pd.to_datetime(features["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    days = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(days / 30.0, 1e-6)


def _exp_lcb(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    arr = np.asarray(pnls, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean
    std = float(arr.std(ddof=0))
    return float(mean - std / (len(arr) ** 0.5))


def _write_markdown(out_path: Path, summary: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Timeframe Sweep From 1m")
    lines.append("")
    lines.append(f"- base_timeframe: `{summary['base_timeframe']}`")
    lines.append(f"- symbols: `{', '.join(summary['symbols'])}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append("")
    lines.append("| timeframe | trade_count | trades/month | PF | expectancy | exp_lcb | maxDD | runtime_seconds |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        lines.append(
            f"| {row['timeframe']} | {float(row['trade_count']):.1f} | {float(row['trades_per_month']):.3f} | "
            f"{float(row['profit_factor']):.6f} | {float(row['expectancy']):.6f} | {float(row['exp_lcb']):.6f} | "
            f"{float(row['max_drawdown']):.6f} | {float(row['runtime_seconds']):.4f} |"
        )
    lines.append("")
    if summary.get("best_by_profit_factor"):
        lines.append(
            f"- best PF timeframe: `{summary['best_by_profit_factor']['timeframe']}` "
            f"({float(summary['best_by_profit_factor']['profit_factor']):.6f})"
        )
    if summary.get("best_by_exp_lcb"):
        lines.append(
            f"- best exp_lcb timeframe: `{summary['best_by_exp_lcb']['timeframe']}` "
            f"({float(summary['best_by_exp_lcb']['exp_lcb']):.6f})"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    tfs = [item.strip().lower() for item in str(args.tfs).split(",") if item.strip()]
    summary = run_timeframe_sweep(
        config=config,
        symbols=symbols,
        base_timeframe=str(args.base_timeframe),
        operational_timeframes=tfs,
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    _write_markdown(args.out_md, summary)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_json}")


if __name__ == "__main__":
    main()
