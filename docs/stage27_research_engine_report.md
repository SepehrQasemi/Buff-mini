# Stage-27 Research Engine Report

## What Changed
- Added rolling-window discovery (3m/6m windows with 1m step).
- Enabled batch score computation in conditional evaluation.
- Reused feature/context frames across windows (compute-once per symbol/timeframe).

## Runtime
- semantic_sample_naive_seconds: `1.821500`
- semantic_sample_batch_seconds: `1.842804`
- rolling_discovery_runtime_seconds: `25.124099`

## Semantic Equivalence Guard
- semantic_hash_equal: `True`
- naive_hash: `6f75d10c058c8a1a`
- batch_hash: `6f75d10c058c8a1a`

## Cache Reuse
- feature_compute_calls: `10`
- rolling_window_evaluations: `10`
- feature_cache_hit_rate_estimate: `0.000000`

## Run Output
- run_id: `20260303_122330_641b29275325_stage27_roll`
- rows: `10`
- used_symbols: `['BTC/USDT', 'ETH/USDT']`
