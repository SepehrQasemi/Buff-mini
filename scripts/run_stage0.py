"""Run Stage-0 baseline strategies and persist artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, stage0_strategies
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.loader import fetch_ohlcv
from buffmini.data.storage import load_parquet, save_parquet
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-0 baselines")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not config["evaluation"]["stage0_enabled"]:
        logger.info("Stage-0 is disabled in config. Exiting.")
        return

    np.random.seed(int(config["search"]["seed"]))
    config_hash = compute_config_hash(config)
    run_id = f"{utc_now_compact()}_{config_hash}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    meta = {
        "run_id": run_id,
        "seed": int(config["search"]["seed"]),
        "config_hash": config_hash,
        "config_path": str(args.config),
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    universe = config["universe"]
    timeframe = universe["timeframe"]
    start = universe["start"]
    end = universe["end"]

    costs = config["costs"]
    risk = config["risk"]

    summary_rows: list[dict[str, float | str]] = []

    for symbol in universe["symbols"]:
        try:
            raw = load_parquet(symbol=symbol, timeframe=timeframe)
            logger.info("Loaded cached data for %s", symbol)
        except FileNotFoundError:
            logger.info("No cached data for %s. Downloading.", symbol)
            raw = fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
            if raw.empty:
                logger.warning("Skipping %s: no data", symbol)
                continue
            save_parquet(frame=raw, symbol=symbol, timeframe=timeframe)

        features = calculate_features(raw)
        strategies = stage0_strategies()

        for strategy in strategies:
            strategy_df = features.copy()
            strategy_df["signal"] = generate_signals(strategy_df, strategy)

            result = run_backtest(
                frame=strategy_df,
                strategy_name=strategy.name,
                symbol=symbol,
                stop_atr_multiple=1.5,
                take_profit_atr_multiple=3.0,
                max_hold_bars=24,
                round_trip_cost_pct=float(costs["round_trip_cost_pct"]),
                slippage_pct=float(costs["slippage_pct"]),
                initial_capital=10_000.0 * float(risk["max_concurrent_positions"]),
            )

            sanitized = strategy.name.lower().replace(" ", "_")
            symbol_key = symbol.replace("/", "-")
            trades_path = run_dir / f"trades_{symbol_key}_{sanitized}.csv"
            equity_path = run_dir / f"equity_{symbol_key}_{sanitized}.csv"

            result.trades.to_csv(trades_path, index=False)
            result.equity_curve.to_csv(equity_path, index=False)

            metrics_row: dict[str, float | str] = {
                "symbol": symbol,
                "strategy": strategy.name,
                **result.metrics,
            }
            summary_rows.append(metrics_row)

            logger.info("%s | %s | metrics=%s", symbol, strategy.name, result.metrics)

    summary = pd.DataFrame(summary_rows)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    if summary.empty:
        logger.warning("No results produced.")
    else:
        print(summary.to_string(index=False))
    logger.info("Saved run artifacts to %s", run_dir)


if __name__ == "__main__":
    main()
