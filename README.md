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

## Stage-9 Data Expansion

Stage-9 adds futures-derived context data (BTC/USDT and ETH/USDT only):

- Funding-rate ingestion + alignment
- Open-interest ingestion + alignment
- Leakage-safe funding/OI interaction features
- Impact analysis reports (statistical bias evidence only)
- Minimal DSL-lite regime selectors for family selection (non-blocking)

Stage-9 is disabled by default for backward compatibility (`data.include_futures_extras: false`).

Run Stage-9 data + analysis:

```bash
python scripts/update_futures_extras.py
python scripts/run_stage9_impact.py
```

Run Stage-9.3 recent OI overlay evidence report (OI masked outside trailing window):

```bash
python scripts/run_stage9_3_overlay_report.py
```

Optional Stage-5 pipeline flag to include futures extras in feature generation:

```bash
python scripts/run_pipeline.py --include-futures-extras
```

Outputs:

- `docs/stage9_report.md`
- `docs/stage9_report_summary.json`
- `docs/stage9_impact_analysis.md`
- `docs/stage9_data_quality.md`
- `docs/stage9_3_recent_oi_overlay.md`
- `docs/stage9_3_recent_oi_overlay_summary.json`

## Stage-10 Engine Upgrade

Stage-10 adds:

- Regime scores + confidence labels
- Expanded signal and exit libraries
- Regime-aware soft activation (sizing multipliers, not hard trade blocking)
- Stage-10.6 refinement: score-only activation input, tighter multiplier clamp, reduced default exits, and sandbox signal ranking
- Baseline vs Stage-10 comparison runner with deterministic artifacts

Run synthetic offline Stage-10:

```bash
python scripts/run_stage10.py --dry-run --seed 42
```

Run Stage-10 on local real data (if `data/raw` exists):

```bash
python scripts/run_stage10.py --seed 42
```

Run Stage-10.6 sandbox ranking (offline-friendly):

```bash
python scripts/run_stage10_sandbox.py --dry-run --seed 42
```

Run Stage-10.7 sandbox ranking on local real data:

```bash
python scripts/run_stage10_sandbox.py --real-data --seed 42
```

Run Stage-10.7 exit A/B isolation:

```bash
python scripts/run_stage10.py --seed 42 --exit-mode compare
```

Outputs:

- `runs/<run_id>_stage10/stage10_summary.json`
- `runs/<run_id>_stage10/stage10_compare.csv`
- `runs/<run_id>_stage10/regime_distribution.csv`
- `runs/<run_id>_stage10/best_candidates.json`
- `docs/stage10_report.md`
- `docs/stage10_report_summary.json`
- `runs/<run_id>_stage10_sandbox/sandbox_rankings.csv`
- `runs/<run_id>_stage10_sandbox/sandbox_summary.json`
- `docs/stage10_6_report.md`
- `docs/stage10_6_report_summary.json`
- `docs/stage10_7_report.md`
- `docs/stage10_7_report_summary.json`

## Stage-11 MTF Engine Capability

Stage-11 adds a config-driven multi-timeframe infrastructure with:

- Causal resample/alignment (`merge_asof` backward-only)
- Deterministic MTF feature-pack caching
- Optional bias/confirm/exit hooks (no-op when disabled)
- Stage-10.7 baseline vs Stage-11 comparison with trade-count guard

Run synthetic offline Stage-11:

```bash
python scripts/run_stage11.py --dry-run --seed 42
```

Run Stage-11 on local real parquet data:

```bash
python scripts/run_stage11.py --seed 42
```

Stage-11.1 presets:

- `configs/presets/stage11_bias.yaml`
- `configs/presets/stage11_confirm.yaml`
- `configs/presets/stage11_bias_confirm.yaml`

Use with:

```bash
python scripts/run_stage11.py --preset configs/presets/stage11_bias.yaml --seed 42
```

Run the full Stage-11.1 effectiveness matrix (baseline + bias + confirm + bias_confirm):

```bash
python scripts/run_stage11_matrix.py --seed 42
```

Outputs:

- `runs/<run_id>_stage11_1/stage11_summary.json`
- `runs/<run_id>_stage11_1/mtf_join_stats.json`
- `runs/<run_id>_stage11_1/regime_distribution.csv`
- `runs/<run_id>_stage11_1/comparison_vs_stage10_7.csv`
- `runs/<run_id>_stage11_1/sizing_stats.json`
- `runs/<run_id>_stage11_1/confirm_stats.json`
- `runs/<run_id>_stage11_1/comparison_vs_baseline.json`
- `docs/stage11_report.md`
- `docs/stage11_report_summary.json`
- `docs/stage11_1_report.md`
- `docs/stage11_1_report_summary.json`

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
