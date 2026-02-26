"""Run Stage-0 baseline strategies and persist artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, stage0_strategies
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.loader import fetch_ohlcv
from buffmini.data.storage import load_parquet, save_parquet
from buffmini.types import ConfigDict
from buffmini.utils.logging import get_logger
from buffmini.utils.time import parse_utc_timestamp, utc_now_compact


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Stage-0 baselines")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Run full Stage-0 pipeline on synthetic data.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional fixed run ID for reproducibility.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Output directory for run artifacts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory for raw parquet data when not using dry-run mode.",
    )
    parser.add_argument(
        "--synthetic-bars",
        type=int,
        default=360,
        help="Number of hourly bars to generate in dry-run mode.",
    )
    parser.add_argument(
        "--stage05",
        action="store_true",
        help="Enable Stage-0.5 filters (ATR volatility gate + optional regime gate).",
    )
    return parser.parse_args()


def run_stage0(
    config: ConfigDict,
    config_path: Path,
    dry_run: bool = False,
    stage05: bool = False,
    run_id: str | None = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    synthetic_bars: int = 360,
) -> Path:
    """Execute Stage-0 for all configured symbols and strategies."""

    if not config["evaluation"]["stage0_enabled"]:
        logger.info("Stage-0 is disabled in config. Exiting.")
        return runs_dir

    seed = int(config["search"]["seed"])
    np.random.seed(seed)
    config_hash = compute_config_hash(config)
    resolved_run_id = run_id or f"{utc_now_compact()}_{config_hash}"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    strategies = stage0_strategies()
    serialized_strategies = [
        {
            "name": strategy.name,
            "entry_rules": strategy.entry_rules,
            "exit_rules": strategy.exit_rules,
            "parameters": strategy.parameters,
        }
        for strategy in strategies
    ]
    with (run_dir / "strategies.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized_strategies, handle, indent=2)

    universe = config["universe"]
    timeframe = universe["timeframe"]
    start = universe["start"]
    end = universe["end"]
    symbols = list(universe["symbols"])

    costs = config["costs"]
    risk = config["risk"]

    summary_rows: list[dict[str, float | str]] = []

    for symbol in symbols:
        raw = _load_symbol_data(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            dry_run=dry_run,
            seed=seed,
            synthetic_bars=synthetic_bars,
            data_dir=data_dir,
        )
        if raw.empty:
            logger.warning("Skipping %s due to empty data.", symbol)
            continue

        features = calculate_features(raw)
        for strategy in strategies:
            strategy_df = features.copy()
            strategy_df["signal"] = generate_signals(strategy_df, strategy, stage05=stage05)

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
                "final_equity": float(result.equity_curve["equity"].iloc[-1]) if not result.equity_curve.empty else 0.0,
                **result.metrics,
            }
            summary_rows.append(metrics_row)

            logger.info("%s | %s | metrics=%s", symbol, strategy.name, result.metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    leaderboard = (
        summary_df.sort_values(
            by=["expectancy", "win_rate", "profit_factor"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        if not summary_df.empty
        else summary_df
    )
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    summary_payload: dict[str, Any] = {
        "run_id": resolved_run_id,
        "config_hash": config_hash,
        "seed": seed,
        "dry_run": dry_run,
        "stage05": stage05,
        "symbol_count": len(symbols),
        "strategy_count": len(strategies),
        "total_combinations": len(symbols) * len(strategies),
        "combinations_executed": int(len(summary_rows)),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    meta = {
        "run_id": resolved_run_id,
        "seed": seed,
        "config_hash": config_hash,
        "config_path": str(config_path),
        "dry_run": dry_run,
        "stage05": stage05,
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if leaderboard.empty:
        logger.warning("No Stage-0 results were produced.")
    else:
        print(leaderboard.to_string(index=False))
    logger.info("Saved run artifacts to %s", run_dir)
    return run_dir


def _load_symbol_data(
    symbol: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    dry_run: bool,
    seed: int,
    synthetic_bars: int,
    data_dir: Path,
) -> pd.DataFrame:
    """Load cached/fetched market data or produce deterministic synthetic bars."""

    if dry_run:
        return _generate_synthetic_ohlcv(symbol=symbol, start=start, bars=synthetic_bars, seed=seed)

    try:
        logger.info("Loading cached data for %s", symbol)
        return load_parquet(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    except FileNotFoundError:
        logger.info("No cached data for %s. Downloading from Binance.", symbol)
        fetched = fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
        if not fetched.empty:
            save_parquet(frame=fetched, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
        return fetched


def _generate_synthetic_ohlcv(symbol: str, start: str | None, bars: int, seed: int) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline smoke tests."""

    bars = max(int(bars), 220)
    symbol_seed = _symbol_seed(seed=seed, symbol=symbol)
    rng = np.random.default_rng(symbol_seed)

    start_ts = parse_utc_timestamp(start) or pd.Timestamp("2024-01-01T00:00:00Z")
    timestamps = pd.date_range(start=start_ts, periods=bars, freq="h", tz="UTC")

    base_price = 100.0 + (symbol_seed % 50)
    log_returns = rng.normal(loc=0.0001, scale=0.008, size=bars)
    close = base_price * np.exp(np.cumsum(log_returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    spread = rng.uniform(0.0005, 0.01, size=bars)
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


def _symbol_seed(seed: int, symbol: str) -> int:
    """Create deterministic per-symbol seed from base seed."""

    digest = hashlib.sha256(f"{seed}:{symbol}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = load_config(args.config)
    run_stage0(
        config=config,
        config_path=args.config,
        dry_run=bool(args.dry_run),
        stage05=bool(args.stage05),
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        synthetic_bars=args.synthetic_bars,
    )


if __name__ == "__main__":
    main()
