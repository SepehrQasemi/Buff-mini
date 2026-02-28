# Stage-9 Report

## What Was Added
- Funding and open-interest derived ingestion/store
- Leakage-safe futures extras features
- Stage-9 impact analysis tables and quality report

## Data Lineage
- BTC/USDT: funding_rows=3458 (2023-01-01T00:00:00+00:00..2026-02-26T08:00:00.006000+00:00), oi_rows=459 (2026-02-07T07:00:00+00:00..2026-02-26T09:00:00+00:00)
- ETH/USDT: funding_rows=3458 (2023-01-01T00:00:00+00:00..2026-02-26T08:00:00.006000+00:00), oi_rows=459 (2026-02-07T07:00:00+00:00..2026-02-26T09:00:00+00:00)

## Leak-Proof Evidence
- features_checked: `36`
- leaks_found: `0`

## Impact Evidence
- ETH/USDT | crowd_short_risk | 72h | median_diff=0.067734 | CI=[0.016392, 0.076472]
- ETH/USDT | crowd_short_risk | 24h | median_diff=-0.022290 | CI=[-0.042118, 0.020032]
- ETH/USDT | funding_extreme_neg | 24h | median_diff=-0.009665 | CI=[-0.015939, 0.000899]

## DSL-Lite Trade-Frequency Guard
- ratio_bounds_ok: `True`
- BTC/USDT: baseline_entries=5908, dsl_entries=5908, ratio=1.0000
- ETH/USDT: baseline_entries=6004, dsl_entries=6004, ratio=1.0000

## Runtime Notes
- impact_analysis_runtime_seconds: `10.562`
- download runtime tracked via scripts/update_futures_extras.py execution
