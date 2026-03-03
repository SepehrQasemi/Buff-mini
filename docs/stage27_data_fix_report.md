# Stage-27 Data Fix Report

## What Changed
- Added strict coverage gating with config controls:
  - `data.coverage.required_years`
  - `data.coverage.min_years_to_run`
  - `data.coverage.fail_if_below_min`
- Added deterministic BTC-only auto-fallback when a non-BTC symbol is below `min_years_to_run`.
- Patched the canonical downloader to:
  - retry with deterministic backoff on rate-limit/network errors
  - backfill missing historical head-range (not only append tail)
  - keep strict dedup/sort/no-future guarantees
- Updated coverage math to use inclusive bar-span so full-window datasets are not undercounted by one bar.

## Coverage Result After Fix
- BTC/USDT 1m coverage: `4.0` years
- ETH/USDT 1m coverage: `4.0` years
- Coverage gate status: `PASS` (no BTC-only fallback required)

## Data Integrity
- Duplicates: `0` for BTC and ETH
- Non-monotonic timestamps: `false` for BTC and ETH
- Gaps detected: `1` per symbol (max gap `81` minutes)
- Canonical + derived data integrity: `PASS`

## Commands Executed
- `python scripts/update_canonical_data.py --symbols BTC/USDT,ETH/USDT --timeframe 1m --years 4`
- `python scripts/build_canonical_timeframes.py --symbols BTC/USDT,ETH/USDT --base 1m --timeframes "5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M"`
- `python scripts/run_stage26_9_full_audit.py`

## Frozen Snapshot
- snapshot_id: `DATA_FROZEN_v1`
- snapshot_file: `data/snapshots/DATA_FROZEN_v1.json`
- snapshot_hash: `c734cebc1e80bf15`

## Notes
- This stage enforces minimum-run coverage with explicit fail/allow behavior in Stage-24/25/26 runners.
- If a future symbol drops below `min_years_to_run`, runs stop unless `--allow-insufficient-data` is set; BTC fallback is recorded when applied.
