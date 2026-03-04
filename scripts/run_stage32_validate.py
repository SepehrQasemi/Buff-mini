"""Run Stage-32 Pareto + nested WF/MC validation for finalists."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage10.evaluate import _build_features
from buffmini.stage31.dsl import evaluate_strategy
from buffmini.stage31.evolve import EvolverConfig, evolve_strategies
from buffmini.stage32.pareto import ParetoConfig, pareto_select
from buffmini.stage32.validate import ValidationConfig, validate_candidates
from buffmini.stage28.window_calendar import generate_window_calendar
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-32 validation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--finalists", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _feature_list(frame: pd.DataFrame) -> list[str]:
    preferred = ["open", "high", "low", "close", "volume", "atr_pct", "ema_20", "ema_50", "ema_200", "rsi_14"]
    out = [c for c in preferred if c in frame.columns]
    return out or [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]


def _candidate_metrics(frame: pd.DataFrame, evolved: pd.DataFrame, symbol: str, timeframe: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    calendar = generate_window_calendar(ts, window_months=3, step_months=1) if not ts.dropna().empty else pd.DataFrame()
    for rec in evolved.to_dict(orient="records"):
        strategy = rec.get("strategy")
        if strategy is None:
            continue
        signal = evaluate_strategy(strategy, frame)
        result = run_backtest(
            frame=frame.assign(signal=signal),
            strategy_name=f"Stage32::{getattr(strategy, 'name', 'cand')}",
            symbol=str(symbol),
            signal_col="signal",
            stop_atr_multiple=float(getattr(strategy, "stop_atr_multiple", 1.5)),
            take_profit_atr_multiple=float(getattr(strategy, "take_profit_atr_multiple", 3.0)),
            max_hold_bars=int(getattr(strategy, "max_hold_bars", 24)),
            round_trip_cost_pct=0.1,
            slippage_pct=0.0005,
            exit_mode=str(getattr(strategy, "exit_mode", "fixed_atr")),
            cost_model_cfg={},
        )
        pf_raw = float(result.metrics.get("profit_factor", 0.0))
        maxdd = float(result.metrics.get("max_drawdown", 0.0))
        pnl = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        exp_lcb = float(np.percentile(pnl, 5)) if pnl.size else 0.0
        repeatability = 0.0
        if not calendar.empty:
            positives = 0
            total = 0
            for w in calendar.to_dict(orient="records"):
                mask = (ts >= pd.to_datetime(w["window_start"], utc=True, errors="coerce")) & (
                    ts < pd.to_datetime(w["window_end"], utc=True, errors="coerce")
                )
                part = frame.loc[mask].reset_index(drop=True)
                if part.shape[0] < 50:
                    continue
                total += 1
                sig_p = evaluate_strategy(strategy, part)
                ret = pd.to_numeric(part.get("close"), errors="coerce").pct_change().fillna(0.0)
                pnl_p = (sig_p.shift(1).fillna(0) * ret).fillna(0.0).sum()
                if float(pnl_p) > 0:
                    positives += 1
            repeatability = float(positives / max(1, total))
        rows.append(
            {
                "candidate_id": str(rec.get("strategy_id", "")),
                "strategy": strategy,
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "exp_lcb": float(exp_lcb),
                "pf_adj": float(np.clip(pf_raw, 0.0, 10.0)),
                "maxdd_p95": float(maxdd),
                "repeatability": float(repeatability),
                "feasibility_score": 1.0,
                "fitness": float(rec.get("fitness", 0.0)),
                "trade_count": int(rec.get("trade_count", 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    seed = int(args.seed)
    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    config_hash = compute_config_hash(cfg)

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': symbols, 'timeframes': timeframes, 'finalists': int(args.finalists), 'dry': bool(args.dry_run), 'cfg': config_hash}, length=12)}"
        "_stage32"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage32"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[pd.DataFrame] = []
    data_hash_parts: dict[str, str] = {}
    resolved_end: list[str] = []
    for timeframe in timeframes:
        fmap = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(timeframe),
            dry_run=bool(args.dry_run),
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in sorted(fmap.items()):
            features = _feature_list(frame)
            evolved = evolve_strategies(
                frame=frame,
                features=features,
                cfg=EvolverConfig(population_size=80, generations=10, elite_count=30, seed=seed),
            )
            cand_metrics = _candidate_metrics(frame, evolved, str(symbol), str(timeframe), seed)
            if not cand_metrics.empty:
                metrics_rows.append(cand_metrics)
            data_cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
            data_hash_parts[f"{symbol}|{timeframe}"] = stable_hash(frame.loc[:, data_cols].to_dict(orient="list"), length=16)
            ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_end.append(ts.max().isoformat())

    all_candidates = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    finalists = pareto_select(all_candidates, cfg=ParetoConfig(top_k=int(max(1, args.finalists))))
    validation_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for (symbol, timeframe), group in finalists.groupby(["symbol", "timeframe"], dropna=False):
        fmap = _build_features(
            config=cfg,
            symbols=[str(symbol)],
            timeframe=str(timeframe),
            dry_run=bool(args.dry_run),
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        frame = fmap.get(str(symbol))
        if frame is None:
            continue
        validated, summary = validate_candidates(
            frame=frame,
            candidates=group,
            symbol=str(symbol),
            timeframe=str(timeframe),
            cfg=ValidationConfig(seed=seed),
        )
        if not validated.empty:
            validation_rows.append(validated)
        summary_rows.append({"symbol": str(symbol), "timeframe": str(timeframe), **summary})

    validated_df = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    validated_df.to_csv(out_dir / "validated.csv", index=False)
    summary = {
        "stage": "32.2",
        "run_id": run_id,
        "seed": seed,
        "symbols": symbols,
        "timeframes": timeframes,
        "finalists_requested": int(args.finalists),
        "finalists_count": int(finalists.shape[0]) if not finalists.empty else 0,
        "wf_executed_pct": float(validated_df["wf_executed"].mean() * 100.0) if not validated_df.empty else 0.0,
        "mc_trigger_pct": float(validated_df["mc_triggered"].mean() * 100.0) if not validated_df.empty else 0.0,
        "validation_summaries": summary_rows,
        "config_hash": config_hash,
        "data_hash": stable_hash(data_hash_parts, length=16),
        "resolved_end_ts": max(resolved_end) if resolved_end else None,
        "runtime_seconds": float(time.perf_counter() - started),
        **snapshot_metadata_from_config(cfg),
    }
    (out_dir / "validated.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"stage32_dir: {out_dir}")


if __name__ == "__main__":
    main()

