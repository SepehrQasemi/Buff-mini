# Stage-3.1 Monte Carlo Full Audit

- Audit date (UTC): 2026-02-27
- Repository scope: `Buff-mini` only
- Target Stage-2 run: `20260227_015806_3cb775eb81a0_stage2`
- Target Stage-3.1 run: `20260227_043640_92029913e884_stage3_1_mc`
- Small consistency rerun: `20260227_103610_111d95f70a65_stage3_1_mc`

## 1) Static Code Audit (YES/NO)

| Check | Verdict | Evidence |
| --- | --- | --- |
| 1) Portfolio trade PnL extracted with EXACT Stage-2 semantics (same costs, no double fee, same entry/exit logic) | YES | `src/buffmini/portfolio/monte_carlo.py` reconstructs candidate bundles through `evaluate_candidate_bundles_for_window(...)`, which calls `_run_candidate_bundle_stage2(...)` -> `builder._run_candidate_bundle(...)` -> `run_backtest(...)` with the same cost inputs from Stage-1 config. Stage-3 only applies portfolio weights to `candidate_pnl`; it does not re-apply fees/slippage. |
| 2) Leverage applied after costs as `pnl_scaled = pnl * leverage` | YES | In `simulate_equity_paths(...)`, sampled trade PnLs (`values`) are already net trade PnLs and are scaled once via `sampled = values[indices] * leverage`. |
| 3) Block bootstrap correctness (contiguous blocks, size in trades, exact final length) | YES | `sample_block_indices(...)` samples contiguous starts, adds fixed offsets `0..block-1`, then truncates to exactly `n_trades` columns. Block size unit is number of trades. |
| 4) Seed propagation deterministic | YES | `np.random.default_rng(seed)` is used with explicit seed; Stage-3 uses deterministic method offsets (`seed + offset`). Added deterministic unit test confirms identical path and summary outputs for same seed. |
| 5) Max drawdown formula correctness | YES | `compute_equity_path_metrics(...)` and vectorized path simulation compute `drawdown = (peak - equity)/peak` with running peaks over equity including initial equity anchor. |
| 6) NaN/inf explicitly prevented in summaries | YES | `summarize_mc(...)` validates finite summary payload (`_payload_has_no_inf_nan`) and raises on invalid values; Stage-3 run summary repeats this check before save. |

## 2) Strict Test Additions

Added: `tests/test_stage3_monte_carlo_audit.py`

- Determinism (same seed => identical paths and summary)
- Block bootstrap structural contiguity
- Max drawdown exact toy-sequence check
- Ruin probability bounds in `[0,1]`
- No NaN/inf in summary payload

Quality gates executed:

- `python -m compileall src scripts tests` -> PASS
- `pytest -q` -> PASS

## 3) Cross-Check vs Stage-2 Baseline

Cross-check artifact: `docs/stage3_1_mc_crosscheck.md`

Result: PASS for all methods (`equal`, `vol`, `corr-min`)

- trade_count reconstructed matches Stage-3 source count
- baseline return is within MC `[p05, p95]`
- baseline maxDD <= MC p95 maxDD
- trade timestamps lie inside Stage-2 holdout range

## 4) Re-run Consistency Check (20k paths vs 5k paths)

Thresholds required:

- return quantile absolute delta `< 0.02`
- maxDD p95 absolute delta `< 0.01`

| Method | abs diff return p05 | abs diff return median | abs diff return p95 | abs diff maxDD p95 | Threshold pass |
| --- | ---: | ---: | ---: | ---: | --- |
| equal | 0.000391 | 0.000341 | 0.000938 | 0.000252 | PASS |
| vol | 0.000531 | 0.001064 | 0.000014 | 0.000406 | PASS |
| corr-min | 0.001090 | 0.000822 | 0.000390 | 0.000411 | PASS |

Interpretation:

- All deltas are far below tolerance bands.
- No severe quantile inconsistency detected.

## 5) Final Verdict

VERDICT: **VERIFIED**

Reasoning:

- Determinism checks pass.
- Stage-2/Stage-3 trade_count and metric cross-checks pass.
- Re-run quantile consistency is well within required tolerances.
- No NaN/inf integrity issues detected in tested summaries.

Warnings: **None**.
