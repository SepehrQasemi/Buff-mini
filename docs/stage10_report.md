# Stage-10 Report

## What Changed
- Stage-10.1 regime scores and labels with confidence
- Stage-10.2 expanded entry signal library (6 families)
- Stage-10.3 expanded exit library
- Stage-10.4 soft regime-aware activation (sizing multipliers only)

## Determinism + Leakage
- determinism: `PASS` (`36032bafe9ac53aac4ee060c`)
- leakage: `PASS` (features_checked=34)

## Before vs After
| metric | baseline | stage10 | delta |
| --- | ---: | ---: | ---: |
| trade_count | 6275.00 | 5627.00 | -648.00 |
| profit_factor | 0.616272 | 0.265562 | -0.350710 |
| expectancy | -7.830390 | -3.518895 | 4.311495 |
| max_drawdown | 0.954500 | 0.990047 | 0.035548 |
| pf_adj | 0.619305 | 0.272031 | -0.347275 |
| exp_lcb | -7.896892 | -3.520591 | 4.376301 |

## Regimes
- distribution (%): `{'TREND': 79.3560400621904, 'RANGE': 0.0, 'VOL_EXPANSION': 16.99750515240265, 'VOL_COMPRESSION': 3.6464547854069496, 'CHOP': 0.0}`
- confidence_median: `0.514710`

## Walkforward V2
- enabled: `True`
- baseline classification: `INSUFFICIENT_DATA`
- stage10 classification: `UNSTABLE`

## Recommendation
- Refine Stage-10 signal/activation parameters before Stage-11

- run_id: `20260228_123104_791590f09a7d_stage10`
- config_hash: `42061eca8bb3`
- data_hash: `1ce4fcff98e4608e`
- seed: `42`
- resolved_end_ts: `2026-02-26T09:00:00+00:00`
