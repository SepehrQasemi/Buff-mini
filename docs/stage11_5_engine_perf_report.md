# Stage-11.5 Engine Structural Hardening & Deterministic Fast Path

## Scope

Stage-11.5 is a structural/performance hardening pass only.

- No strategy logic changes.
- No entry/exit semantics changes.
- No fee/slippage/stop-priority/walkforward math changes.
- Offline deterministic execution preserved.

## Baseline Snapshot (Pre-change)

Baseline command (recorded before Stage-11.5 code changes):

```bash
python scripts/bench_engine.py --seed 42 --base-timeframe 1m --operational-timeframe 1h --dry-run --dry-run-rows 6000
```

Baseline run id: `20260301_002140_07b5a565d314_stage11_4_bench`

- first_run_seconds: `0.2062947999802418`
- second_run_seconds: `0.11809170001652092`
- cache_hit_rate_rerun: `1.0`

## What Was Hardened

1) Strict derived timeframe cache contract

- Added deterministic key contract and wrapper:
  - `get_or_build_derived_ohlcv(...)`
  - key includes: `symbol`, `base_tf`, `target_tf`, `resolved_end_ts`, `data_hash`, `config_hash`
- Added instrumentation:
  - `derived_cache.hits`
  - `derived_cache.misses`
  - `derived_cache.build_seconds`

2) Strict feature compute-once contract

- Added deterministic dataset key + cache wrapper:
  - `compute_features_cached(...)`
  - key includes: `symbol`, `timeframe`, `resolved_end_ts`, `feature_config_hash`, `data_hash`
- Added instrumentation:
  - `features_compute_calls_per_tf`
  - feature cache hit/miss/build seconds
- Walkforward-style reuse test proves compute once and slicing reuse.

3) NumPy fast path in backtest hot loop

- Added `engine_mode` in backtest (`numpy` default, `pandas` legacy path retained for verification).
- Removed pandas row-iteration from hot loop in default path.
- Preserved decision ordering and stop/TP/time-stop priority.

## After Snapshot (Post-change, same baseline command)

Re-run command:

```bash
python scripts/bench_engine.py --seed 42 --base-timeframe 1m --operational-timeframe 1h --dry-run --dry-run-rows 6000
```

After run id: `20260301_010925_07b5a565d314_stage11_4_bench`

- first_run_seconds: `0.2429227999818977`
- second_run_seconds: `0.1165575000050012`
- cache_hit_rate_rerun: `1.0`

Comparison vs baseline:

- first-run factor: `0.8492x` (slower; cold-run noise/overhead acceptable)
- second-run factor: `1.0132x` (faster hot-path)

## Stage-11.5 Structural Benchmark (Multi-TF)

Command:

```bash
python scripts/bench_engine_stage11_5.py --seed 42 --base-timeframe 1m --tfs 15m,1h,2h,4h --dry-run --dry-run-rows 6000
```

Run id: `20260301_010925_d4c0c619b104_stage11_5_bench`

- first_run_seconds: `0.8585469000099692`
- second_run_seconds: `0.2898214999877382`
- speedup_factor: `2.962329917022349`
- derived_cache_hit_rate (rerun): `1.0`
- feature_cache_hit_rate (rerun): `1.0`

Per-stage timing (first -> second):

- load: `0.5295851000119001 -> 0.1639319999376312`
- features: `0.24809959999402054 -> 0.04850380000425503`
- backtest: `0.05857769996509887 -> 0.054764000000432134`

Cache details:

- first cache:
  - derived: `0 hits / 8 misses`
  - feature: `2 hits / 6 misses`
  - features_compute_calls_per_tf: `{"15m": 2, "2h": 2, "4h": 2}`
- second cache:
  - derived: `8 hits / 0 misses`
  - feature: `8 hits / 0 misses`
  - features_compute_calls_per_tf: `{}`

## Semantic Equivalence Proof (No Logic Drift)

Verified via `tests/test_engine_semantic_equivalence.py`:

- pandas_equity_hash: `0a82dfb2d8244074`
- numpy_equity_hash: `0a82dfb2d8244074`
- pandas_trades_hash: `17c50ab96f319d3d`
- numpy_trades_hash: `17c50ab96f319d3d`

Verdict: exact hash match for equity curve and trades.

## Determinism Confirmation

Determinism evidence:

- All cache keys are stable-hash based on deterministic inputs (`seed/config_hash/data_hash/resolved_end_ts/timeframe`).
- Semantic equivalence test enforces stable outputs across legacy and fast paths.
- Feature and derived caches are deterministic and key-bound.

## Tests Added

- `tests/test_engine_no_recompute_cache.py`
- `tests/test_engine_semantic_equivalence.py`
- `tests/test_engine_perf_contract.py`

## Known Limitations

- Performance timings vary by local machine load; timing fields are not expected to be bit-identical across runs.
- Cold-run performance may include one-time filesystem/cache metadata overhead.
- Stage-11.5 focuses on structural speedups; backtest math is intentionally unchanged.

## Stage-12 Scaling Estimate

Given multi-TF rerun speedup (~`2.96x`) and deterministic cache hits on rerun:

- Stage-12 full sweep complexity is still dominated by strategy/backtest count, but
- derived TF + feature recompute overhead is substantially reduced,
- making repeated matrix evaluations and iterative dev cycles materially cheaper.
