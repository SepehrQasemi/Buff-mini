# Stage-35 Report

## Summary
- head_commit: `cea6093`
- run_id: `20260306_002542_12d9e1dca8f2_stage35`
- coinapi_enabled: `False`
- dry_run: `False`
- status: `INSUFFICIENT_COVERAGE`

## Endpoints
- requested: `['funding_rates', 'open_interest', 'liquidations']`
- download_attempted: `False`

## Coverage
- required_years: `2.0`
- BTC/USDT:
  - funding_rates: years=0.011, samples=3, range=2026-01-01T00:00:00+00:00..2026-01-05T00:00:00+00:00
  - liquidations: years=0.000, samples=0, range=None..None
  - open_interest: years=0.000, samples=0, range=None..None
- ETH/USDT:
  - funding_rates: years=0.000, samples=0, range=None..None
  - liquidations: years=0.000, samples=0, range=None..None
  - open_interest: years=0.000, samples=0, range=None..None

## Usage
- total_requests: `1682`
- total_success: `842`
- total_fail: `840`

## ML Trigger
- ml_executed: `False`
- reason: `insufficient_coverage`

## Blocking Reasons
- BTC/USDT/funding_rates: years=0.011, missing_days~726.5
- BTC/USDT/open_interest: years=0.000, missing_days~730.5
- ETH/USDT/funding_rates: years=0.000, missing_days~730.5
- ETH/USDT/open_interest: years=0.000, missing_days~730.5

## Storage
- raw_mb: `0.000`
- canonical_mb: `0.012`
- meta_mb: `1.043`
- total_mb: `1.055`
