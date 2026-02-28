# Stage-10 Report

## Scope
- Stage-10.1 regime scoring with confidence
- Stage-10.2 signal library expansion (6 families)
- Stage-10.3 exit library expansion
- Stage-10.4 regime-aware soft activation (sizing multipliers only; no hard trade gating)
- Stage-10.5 evaluation harness with Stage-8 walkforward_v2 compatibility and cost_model support

## Run Provenance
- Dry-run (synthetic, offline): `20260228_111937_2d4caf91e389_stage10`
- Real local data: `20260228_112303_c3a6713a904d_stage10`
- Seed: `42`
- Config hash: `d38170f35d6f`

## Determinism + Leakage
- Determinism status: `PASS`
- Leakage status: `PASS` (`features_checked=34`, `leaks_found=0`)

## Dry-Run Evidence (Synthetic Offline)
| metric | baseline | stage10 | delta |
| --- | ---: | ---: | ---: |
| trade_count | 519.00 | 450.00 | -69.00 |
| profit_factor | 0.618459 | 0.455899 | -0.162560 |
| expectancy | -32.934793 | -25.574440 | 7.360353 |
| max_drawdown | 0.441529 | 0.601970 | 0.160442 |
| pf_adj | 0.651986 | 0.510309 | -0.141677 |
| exp_lcb | -34.028256 | -25.625012 | 8.403244 |

## Real-Data Evidence (Local)
| metric | baseline | stage10 | delta |
| --- | ---: | ---: | ---: |
| trade_count | 6275.00 | 2923.00 | -3352.00 |
| profit_factor | 0.616272 | 0.322475 | -0.293797 |
| expectancy | -7.830390 | -3.042735 | 4.787655 |
| max_drawdown | 0.954500 | 0.970305 | 0.015806 |
| pf_adj | 0.619305 | 0.333869 | -0.285436 |
| exp_lcb | -7.896892 | -4.974870 | 2.922021 |

## Regime Distribution (Real Run, %)
- `TREND`: 79.3560
- `RANGE`: 0.0000
- `VOL_EXPANSION`: 16.9975
- `VOL_COMPRESSION`: 3.6465
- `CHOP`: 0.0000
- `confidence_median`: 0.514710

## Walkforward V2 (Real Run)
- Enabled: `true`
- Baseline classification: `INSUFFICIENT_DATA`
- Stage-10 classification: `UNSTABLE`

## Cost Model Context
- Stage-10 runs used `cost_mode=v2` (cost_model v2 active).
- Baseline in this report means pre-Stage10 strategy logic under the same run context.

## Recommendation
- `Refine Stage-10 signal/activation parameters before Stage-11`

## Notes
- No profitability guarantee is implied.
- Stage-10 in this revision improves structure and diagnostics, but stability is not yet sufficient for Stage-11 expansion.
