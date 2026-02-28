# Stage-9 Report

## What Was Added
- Funding and open-interest derived ingestion/store
- Leakage-safe futures extras features
- Stage-9 impact analysis tables and quality report

## Data Lineage
- BTC/USDT: funding_rows=3458 (2023-01-01T00:00:00+00:00..2026-02-26T08:00:00.006000+00:00), oi_rows=703 (2026-01-28T03:00:00+00:00..2026-02-26T09:00:00+00:00)
- ETH/USDT: funding_rows=3458 (2023-01-01T00:00:00+00:00..2026-02-26T08:00:00.006000+00:00), oi_rows=703 (2026-01-28T03:00:00+00:00..2026-02-26T09:00:00+00:00)

## Leak-Proof Evidence
- features_checked: `36`
- leaks_found: `0`

## Impact Evidence
- ETH/USDT | crowd_short_risk | 72h | median_diff=-0.155981 | CI=[-0.175131, 0.016671]
- ETH/USDT | crowd_short_risk | 24h | median_diff=-0.028259 | CI=[-0.046314, -0.020978]
- ETH/USDT | crowd_long_risk | 24h | median_diff=0.013750 | CI=[-0.000513, 0.016256]

## DSL-Lite Trade-Frequency Guard
- ratio_bounds_ok: `True`
- BTC/USDT: baseline_entries=5908, dsl_entries=5908, ratio=1.0000
- ETH/USDT: baseline_entries=6004, dsl_entries=6004, ratio=1.0000

## Runtime Notes
- impact_analysis_runtime_seconds: `13.923`
- download runtime tracked via scripts/update_futures_extras.py execution
