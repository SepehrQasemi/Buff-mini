# Stage-11.55 Engine Hardening Report

## Scope

Stage-11.55 is structural hardening only.

- No trading rule changes.
- No entry/exit/cost/stop/walkforward semantic changes.
- Deterministic behavior preserved.

## 1) Root Cause: Cold-Run Double Compute Signal

Observed in prior Stage-11.5 benchmark artifacts:

- Run: `runs/20260301_010925_d4c0c619b104_stage11_5_bench/perf_meta.json`
- `first_cache.features_compute_calls_per_tf` showed multi-count (`15m:2`, `2h:2`, `4h:2`).

Root causes:

1. `features_compute_calls_per_tf` was aggregated by timeframe across symbols, so two symbols appeared as "double compute" per TF.
2. Benchmark cold-run was not always isolated from pre-existing feature cache files, which blurred true cold-run behavior.

Fixes:

- Added explicit in-process dedup boundary (`FeatureComputeSession`) so dataset-key requests are memoized per run.
- Enforced deterministic dataset-key contract: `(symbol, timeframe, resolved_end_ts, feature_config_hash, data_hash)`.
- Added benchmark isolation controls (`--feature-cache-dir`) and explicit cold/rerun feature-call reporting.

## 2) Before/After Compute Calls Per TF

Before (prior artifact, two-symbol aggregated counting):

- `{"15m": 2, "2h": 2, "4h": 2}` (with 1h warm-hit in that artifact)

After (isolated cold-run, single-symbol deterministic contract):

Command:

```bash
python scripts/bench_engine_stage11_5.py \
  --seed 42 \
  --symbols BTC/USDT \
  --base-timeframe 1m \
  --tfs 15m,1h,2h,4h \
  --dry-run \
  --dry-run-rows 6000 \
  --data-dir data/stage11_55_tmp/raw_after \
  --derived-dir data/stage11_55_tmp/derived_after \
  --feature-cache-dir data/stage11_55_tmp/features_after
```

Run: `20260301_021502_b6b5fac36c47_stage11_5_bench`

- cold_run_feature_calls_per_tf: `{"15m": 1, "1h": 1, "2h": 1, "4h": 1}`
- rerun_feature_calls_per_tf: `{}`
- derived_cache_hit_rate (rerun): `1.0`

## 3) Cold-run vs Rerun Bench Numbers

From `runs/20260301_021502_b6b5fac36c47_stage11_5_bench/perf_meta.json`:

- first_run_seconds: `0.3939`
- second_run_seconds: `0.1545`
- speedup_factor: `2.55x`
- feature_cache_hit_rate (rerun): `1.0`
- derived_cache_hit_rate (rerun): `1.0`

## 4) Stage-12 Preflight Runtime Estimate

Command:

```bash
python scripts/estimate_stage12.py \
  --bench-json runs/20260301_021502_b6b5fac36c47_stage11_5_bench/perf_meta.json
```

Output summary:

- estimated_total_seconds: `8.06`
- estimated_total_minutes: `0.13`
- recommendation: `safe`

## 5) Cache Footprint Guard

Implemented cache registry + LRU enforcement:

- Registry metadata per entry:
  - timestamp
  - symbol
  - timeframe
  - resolved_end_ts
  - size_bytes
- Config defaults:
  - `evaluation.stage11_55.cache.max_entries_per_tf: 5`
  - `evaluation.stage11_55.cache.max_total_mb: 2048`
- On cache write:
  - stale entries are sanitized
  - per-timeframe cap applied first
  - total-size cap applied second
  - oldest entries removed deterministically by `(last_access_at, timestamp, key)`

Validation:

- `tests/test_cache_lru_policy.py`

## 6) Semantic Equivalence & Determinism

Semantic equivalence remains PASS:

- `tests/test_engine_semantic_equivalence.py`
- Equity hash match and trades hash match between `engine_mode=pandas` and `engine_mode=numpy`.

Determinism:

- Stable cache keys include data/config/end timestamp factors.
- Same seed + same config + same local data yields identical outputs.

## 7) Test Evidence

Added/updated tests:

- `tests/test_engine_cold_run_single_feature_compute.py`
- `tests/test_engine_no_double_compute_regression.py`
- `tests/test_engine_perf_contract.py`
- `tests/test_stage12_estimator_contract.py`
- `tests/test_cache_lru_policy.py`

Current suite status:

- `226 passed`

## 8) Stage-12 Readiness Conclusion

Readiness status: **PASS**

- Cold-run feature calls are bounded and deterministic (`1` per TF in isolated single-symbol cold-run).
- Rerun behavior is cache-hit dominant (`feature + derived hit rate = 1.0`).
- Cache growth is bounded by configurable LRU limits.
- Preflight estimator is available for runtime budgeting.

No trading semantics were changed in Stage-11.55.
