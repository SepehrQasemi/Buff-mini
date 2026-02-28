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
python scripts/run_stage0.py --dry-run
streamlit run src/buffmini/ui/app.py
```

One-click local launch (no CLI after clone):

- Windows: double-click `start_buffmini.bat`
- macOS/Linux: `./start_buffmini.sh`
- Details: [docs/quick_start_one_click.md](docs/quick_start_one_click.md)

For offline validation (no exchange download), use `python scripts/run_stage0.py --dry-run`.

Config note: `costs.round_trip_cost_pct` is interpreted as a percent value (`0.1` means `0.1%`, `1.0` means `1.0%`).

## Stage-0 Purpose

Stage-0 is a feasibility gate that runs three transparent baseline strategies on shared data and identical execution rules. It is intended to test whether any signal survives basic costs and risk constraints before any strategy discovery workflow is considered.

No guarantee of profitability. Designed to prevent overfitting.

`scripts/run_discovery.py` is a placeholder in MVP Phase 1 and does not run discovery generation yet.

## Stage-4 Execution Layer

Stage-4 converts selected research outputs into an execution-ready specification with configurable policy and risk controls.

Supported execution modes:

- `net`
- `hedge`
- `isolated`

Generate trading spec from existing Stage-2/Stage-3.3 artifacts:

```bash
python scripts/run_stage4_spec.py \
  --stage2-run-id 20260227_015806_3cb775eb81a0_stage2 \
  --stage3-3-run-id 20260227_113410_aca9cf2325a2_stage3_3_selector
```

Run offline paper-execution simulation:

```bash
python scripts/run_stage4_simulate.py \
  --stage2-run-id 20260227_015806_3cb775eb81a0_stage2 \
  --stage3-3-run-id 20260227_113410_aca9cf2325a2_stage3_3_selector \
  --days 90
```

Outputs:

- `runs/<timestamp>_*_stage4/spec/trading_spec.md`
- `runs/<timestamp>_*_stage4/spec/paper_trading_checklist.md`
- `runs/<timestamp>_*_stage4/policy_snapshot.json`
- `runs/<timestamp>_*_stage4_sim/` execution diagnostics

## Stage-5 Product UI

Stage-5 adds a product-style Streamlit workflow with:

- Strategy Lab (safe one-page runner with presets)
- Run Monitor (live stage progress/logs + cancel + reconnect)
- Results Studio (artifact-driven summary/charts/reports + library export)
- Strategy Library (save/reuse compact strategy packages)
- Run Compare (side-by-side run analysis)
- Paper Trading Playback (bar-by-bar UI from bundled artifacts)

Start UI:

```bash
streamlit run src/buffmini/ui/app.py
```

or use the Stage-5.7 one-click launcher:

- Windows: `start_buffmini.bat`
- macOS/Linux: `start_buffmini.sh`

Run Stage-5 pipeline from CLI:

```bash
python scripts/run_pipeline.py \
  --symbols BTC/USDT,ETH/USDT \
  --timeframe 1h \
  --window-months 12 \
  --mode quick \
  --execution-mode net \
  --seed 42
```

Export any completed pipeline run into `library/`:

```bash
python scripts/export_to_library.py --run-id <pipeline_run_id>
```

Super Run: enable auto-save in Strategy Lab, then Run Monitor exports to library automatically on successful completion.

## Stage-6 Edge Amplification

Stage-6 adds deterministic, no-lookahead execution overlays on top of the existing validated pipeline:

- Regime classifier (`TREND`, `RANGE`, `VOL_EXPANSION`)
- Confidence-weighted component sizing
- Regime-aware dynamic leverage with conservative clipping

Run an offline baseline-vs-Stage-6 comparison:

```bash
python scripts/run_stage6_compare.py --offline --seed 42 --window-months 3
```

Outputs:

- `runs/<stage6_run_id>/stage6_compare/stage6_compare_report.md`
- `runs/<stage6_run_id>/stage6_compare/stage6_compare_summary.json`
- `docs/stage6_report.md`

## Stage-8 Validation Foundation

Stage-8 strengthens research integrity with deterministic validation layers:

- Stage-8.1: standardized multi-window walk-forward (`train/holdout/forward`) with robust stats and stability classification.
- Stage-8.2: cost model v2 with volatility-aware slippage, spread proxy, and deterministic execution delay.
- Stage-8.3: automated future-leakage harness across all registered features.

Run Stage-8 offline checks:

```bash
python scripts/run_stage8_walkforward.py --seed 42 --rows 10080
python scripts/run_stage8_cost_sensitivity.py
```

Outputs:

- `runs/<run_id>_stage8_wf/` walk-forward artifacts
- `docs/stage8_cost_sensitivity.md`
- `docs/stage8_cost_sensitivity.json`
- `docs/stage8_report.md`
- `docs/stage8_report_summary.json`

## Output Structure

- Raw market data: `data/raw/*.parquet`
- Run artifacts: `runs/<timestamp>_<config_hash>/`
  - `config.yaml`
  - `summary.json`
  - `leaderboard.csv`
  - `strategies.json`
  - `meta.json`
  - `summary.csv`
  - strategy-level `trades_*.csv` and `equity_*.csv`
