# Stage-28.10 Real-Data Checks

## Commands Executed
- `python scripts/audit_data_coverage.py --symbols BTC/USDT --base-timeframe 1m --required-years 4`
- `python scripts/run_stage28.py --seed 42 --symbols BTC/USDT --timeframes 1h,4h --windows 3m,6m --step-months 1 --mode research`
- `python scripts/run_stage28.py --seed 42 --symbols BTC/USDT --timeframes 1h,4h --windows 3m,6m --step-months 1 --mode live`

## Data Coverage
- BTC coverage years: `4.000000`
- Snapshot ID: `DATA_FROZEN_v1`
- Snapshot hash: `c734cebc1e80bf15`
- Canonical files present: `data/canonical/binance/BTC-USDT/1h.parquet`, `data/canonical/binance/BTC-USDT/4h.parquet`

## Evidence Pointers
- Research summary: `runs/20260304_204309_3c3acf70f8a3_stage28/stage28/summary.json`
- Live summary: `runs/20260304_205310_31b5c28752ff_stage28/stage28/summary.json`
- Window calendar: `runs/20260304_204309_3c3acf70f8a3_stage28/stage28/window_calendar.csv`
- Funnel summary: `runs/20260304_204309_3c3acf70f8a3_stage28/stage28/funnel_summary.json`
- Usability trace: `runs/20260304_204309_3c3acf70f8a3_stage28/stage28/usability_trace.csv`
- Policy output: `runs/20260304_204309_3c3acf70f8a3_stage28/stage28/policy.json`
