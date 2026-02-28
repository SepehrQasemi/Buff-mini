# Stage-11 Report

## What Stage-11 Adds
- Config-driven MTF spec/layers with causal merge_asof alignment
- Optional bias/confirm/exit hooks with no-op compatibility
- Deterministic MTF feature cache keyed by data/config

## Causality + Leakage
- causality: `PASS`
- leakage: `PASS` (features_checked=36)

## Cache
- enabled: `False`
- hits/misses: `0/0`
- hit_rate: `0.000000`

## Baseline vs Stage-11
| variant | trade_count | PF | expectancy | maxDD | exp_lcb |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_stage10_7 | 2919.00 | 0.261619 | -3.074422 | 0.985551 | -5.175793 |
| stage11_bias_only | 2919.00 | 0.261619 | -3.074422 | 0.985551 | -5.175793 |
| stage11_with_confirm | 2919.00 | 0.261619 | -3.074422 | 0.985551 | -5.175793 |

## Trade Count Guard
- pass: `True`
- observed_drop_pct: `0.000000`
- max_drop_pct: `15.00`

## Walkforward
- classification: `UNSTABLE -> UNSTABLE`
- usable_windows: `16 -> 16`

## Final Verdict
- NEUTRAL

- run_id: `20260228_151921_a544a242e95f_stage11`
- config_hash: `daeee98ef8b9`
- data_hash: `1ce4fcff98e4608e`
- seed: `42`
- determinism_signature: `d67ca901eba7705a070d8906`
