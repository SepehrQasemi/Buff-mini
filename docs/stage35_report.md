# Stage-35 Report

## Summary
- head_commit: `d4da962`
- run_id: `20260305_125344_2b45796557c7_stage35`
- coinapi_enabled: `False`
- dry_run: `True`
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
- total_requests: `0`
- total_success: `0`
- total_fail: `0`

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
- canonical_mb: `0.004`
- meta_mb: `0.001`
- total_mb: `0.005`
