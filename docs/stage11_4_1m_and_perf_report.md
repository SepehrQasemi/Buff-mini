# Stage-11.4 1m Data + Performance Report

## Scope

Stage-11.4 adds deterministic 1-minute data support, exact higher-timeframe derivation from 1m, and cache-driven performance optimizations without changing trading logic.

## What Changed

- 1m ingestion and incremental update support via `scripts/update_data.py --timeframe 1m` (plus `scripts/update_data_1m.py` wrapper).
- Config additions:
  - `universe.base_timeframe`
  - `universe.operational_timeframe`
  - `universe.htf_timeframes`
  - `data.resample_source`
  - `data.partial_last_bucket`
  - `data.feature_cache.enabled`
- Deterministic OHLCV resampling module:
  - `src/buffmini/data/resample.py`
- Derived timeframe/data hash caching:
  - `src/buffmini/data/cache.py`
  - integrated through `ParquetStore` in `src/buffmini/data/store.py`
- Performance benchmark runner:
  - `scripts/bench_engine.py`
- Timeframe sweep from 1m runner:
  - `scripts/run_timeframe_sweep.py`

## Data Correctness (1m)

Source update command run:

```bash
python scripts/update_data.py --timeframe 1m --symbols BTC/USDT,ETH/USDT --start "2025-01-01"
```

Observed local 1m parquet quality:

| Symbol | Rows | First TS (UTC) | Last TS (UTC) | Duplicates | Monotonic | Gaps > 60s |
|---|---:|---|---|---:|---|---:|
| BTC/USDT | 610473 | 2025-01-01T00:00:00+00:00 | 2026-02-28T22:32:00+00:00 | 0 | true | 0 |
| ETH/USDT | 610477 | 2025-01-01T00:00:00+00:00 | 2026-02-28T22:36:00+00:00 | 0 | true | 0 |

## Resample Correctness + Causality Guarantees

Implemented and validated:

- Exact OHLCV aggregation:
  - `open=first`, `high=max`, `low=min`, `close=last`, `volume=sum`
- UTC-epoch-aligned buckets for supported derived timeframes:
  - `5m, 15m, 30m, 1h, 2h, 4h, 1d`
- Incomplete last bucket dropped by default (`data.partial_last_bucket=false`)
- Causal guarantee:
  - aggregated bar at time `T` uses only source rows in its own bucket
  - no forward leakage into earlier buckets

Evidence from tests:

- `tests/test_resample_ohlcv_exact.py`
- `tests/test_resample_is_causal.py`
- `tests/test_data_1m_schema.py`

## Determinism + Cache Effectiveness

Benchmark command run:

```bash
python scripts/bench_engine.py --seed 42 --base-timeframe 1m --operational-timeframe 1h
```

Run id: `20260228_223633_07b5a565d314_stage11_4_bench`

Key results:

- First run total: `3.2246s`
- Second run total: `1.6112s`
- Combined cache hit rate on rerun: `1.0000`

Stage timing (first -> second):

- Load: `1.1227s -> 0.3916s`
- Features: `0.9285s -> 0.0398s`
- Backtest: `1.1734s -> 1.1798s` (expected near-constant; compute path unchanged)

Artifacts:

- `runs/20260228_223633_07b5a565d314_stage11_4_bench/perf_profile.json`

## Pipeline Regression Safety (Legacy 1h)

Legacy-mode regression guard is covered by:

- `tests/test_pipeline_regression_1h_unchanged.py`

Behavior guarantee:

- When `universe.base_timeframe` is absent or equals operational timeframe, loader behavior remains direct and 1h workflow metrics remain stable across repeated runs.

## Timeframe Sweep From 1m

Sweep command run:

```bash
python scripts/run_timeframe_sweep.py --seed 42 --base-timeframe 1m --tfs 15m,30m,1h,2h,4h
```

Outputs:

- `docs/timeframe_sweep_from_1m.md`
- `docs/timeframe_sweep_from_1m.json`

Observed best slices:

- Best PF: `2h` (`1.103170`)
- Best exp_lcb: `1h` (`-5.895416`, least negative among tested TFS)

## Added Test Coverage

- `tests/test_data_1m_schema.py`
- `tests/test_resample_ohlcv_exact.py`
- `tests/test_resample_is_causal.py`
- `tests/test_cache_hit_rate.py`
- `tests/test_pipeline_regression_1h_unchanged.py`
- `tests/test_timeframe_sweep_from_1m.py`

## Usage

Update 1m data:

```bash
python scripts/update_data.py --timeframe 1m --symbols BTC/USDT,ETH/USDT
```

Benchmark cache-aware engine path:

```bash
python scripts/bench_engine.py --seed 42 --base-timeframe 1m --operational-timeframe 1h
```

Run operational timeframe sweep from 1m:

```bash
python scripts/run_timeframe_sweep.py --seed 42 --base-timeframe 1m --tfs 15m,30m,1h,2h,4h
```

## Known Limits

- Benchmark currently profiles data->features->baseline-backtest path and does not include full Stage-11 walkforward loops.
- Sweep metrics are strategy-dependent and intended as engineering validation, not profitability claims.
- Cache files are local artifacts; cache warmup/hit ratios depend on unchanged data/config hashes.
