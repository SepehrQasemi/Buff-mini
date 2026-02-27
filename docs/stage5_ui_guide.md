# Stage-5 UI Guide

Stage-5 provides a product-style Streamlit workflow for running and reviewing Buff-mini pipelines using local artifacts.

## Run UI

```bash
streamlit run src/buffmini/ui/app.py
```

## Pages

- `20_strategy_lab.py`
: One-page runner with presets (`quick`/`full`), window selection, fees, execution mode, and seed.
- `21_run_monitor.py`
: Live monitor for the active run with stage timeline, progress, counters, CPU/RAM, log tail, cancel, and reconnect.
- `22_results_studio.py`
: Artifact-driven result analysis (summary, charts, trade map, exposure/risk, report viewer) and library export.
- `23_strategy_library.py`
: Browse reusable strategy cards, preview specs, and load params back into Strategy Lab.
- `24_run_compare.py`
: Side-by-side run comparison with metrics, equity overlays, and leverage-curve overlays.
- `25_paper_trading.py`
: Bar-by-bar paper trading playback from `ui_bundle/playback_state.csv`.

## One-Active-Run Lock

A single-run lock is enforced through `runs/_active_run.json`:

- Run start acquires lock (`run_id`, `pid`, `command`, `status`).
- If process is dead, stale lock is cleared automatically.
- New runs are blocked while an active PID is alive.
- Monitor page reconnects to active run from lock metadata.

## Safe Execution

UI launch helpers only allow these scripts:

- `scripts/run_pipeline.py`
- `scripts/run_stage4_spec.py`
- `scripts/run_stage4_simulate.py`
- `scripts/export_to_library.py`

Execution is always via argument lists (`subprocess.Popen(..., shell=False)`), with parameter validation and enum checks.

## Strategy Library

Library path:

- `library/index.json`
- `library/strategies/<strategy_id>/`

Exports copy only compact metadata/docs:

- `strategy_card.json`
- `strategy_spec.md`
- `paper_trading_checklist.md`
- `params.json`
- `origin.json`
- optional `weights.csv`

Large run artifacts are not copied to the library.

## Super Run Auto-save

In Strategy Lab, enable `Auto-save best result to Library`.
When the run finishes, Run Monitor automatically executes `scripts/export_to_library.py` and stores the resulting `strategy_id`.

## Playback Contract

`ui_bundle/playback_state.csv` columns:

- `timestamp`
- `symbol`
- `action` (`open|close|hold`)
- `exposure`
- `reason`
- `equity`

## Offline Safety Notes

- UI display is artifact-driven from `runs/` and `docs/`.
- No market-data fetch is performed by Stage-5 pages.
- Pipeline validates required local parquet files first and fails fast if missing.
