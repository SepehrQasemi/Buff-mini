# Buff-mini

Auto Crypto Strategy Discovery Engine (MVP Phase 1).

## What This Phase Builds

- Reproducible configuration system
- Binance 1h OHLCV data ingestion via `ccxt`
- Parquet data storage layer
- Feature pipeline (EMA, RSI, ATR, Donchian)
- Minimal long/short backtest engine with ATR exits and costs
- Stage-0 baseline strategy feasibility run
- CLI scripts and Streamlit skeleton UI
- CI workflow and unit tests

## Architecture Overview

- `configs/`: default and preset YAML configs
- `src/buffmini/config.py`: config loading, validation, hashing
- `src/buffmini/data/`: loader, storage, features
- `src/buffmini/backtest/`: costs, metrics, engine
- `src/buffmini/baselines/stage0.py`: baseline strategies
- `scripts/`: CLI entry scripts for data + Stage-0 runs
- `src/buffmini/ui/`: Streamlit app skeleton and pages
- `tests/`: config, backtest, and no-lookahead checks

## Quickstart

```bash
pip install -e .
python scripts/update_data.py
python scripts/run_stage0.py
streamlit run src/buffmini/ui/app.py
```

## Stage-0 Purpose

Stage-0 is a feasibility gate that runs three transparent baseline strategies on shared data and identical execution rules. It is intended to test whether any signal survives basic costs and risk constraints before any strategy discovery workflow is considered.

No guarantee of profitability. Designed to prevent overfitting.

## Output Structure

- Raw market data: `data/raw/*.parquet`
- Run artifacts: `runs/<timestamp>_<config_hash>/`
  - `config.yaml`
  - `meta.json`
  - `summary.csv`
  - strategy-level `trades_*.csv` and `equity_*.csv`
