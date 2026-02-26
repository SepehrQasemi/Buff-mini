# Stage-1 Real Data Report

- run_id: `20260226_120451_fa8f0e044192_stage1`
- split_mode: `60_20_20`
- recent_weight: `2.0`
- min_holdout_trades: `50`
- round_trip_cost_pct: `0.1`

## Top 5 Candidates
| rank | strategy | gating | exit | val_trades | hold_trades | PF_val | PF_hold | exp_val | exp_hold | max_dd_combined | return_hold | CAGR_hold | rejected |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Trend Pullback | vol+regime | fixed_atr | 0 | 2 | 0.0000 | 10.0000 | 0.0000 | 298.2824 | 0.0005 | 0.0298 | 0.0477 | yes |
| 2 | Trend Pullback | vol | fixed_atr | 0 | 1 | 0.0000 | 10.0000 | 0.0000 | 132.9252 | 0.0003 | 0.0133 | 0.0211 | yes |
| 3 | Trend Pullback | vol+regime | fixed_atr | 0 | 1 | 0.0000 | 10.0000 | 0.0000 | 100.5105 | 0.0003 | 0.0101 | 0.0160 | yes |
| 4 | Trend Pullback | vol+regime | fixed_atr | 2 | 6 | 10.0000 | 10.0000 | 42.0566 | 120.8643 | 0.0005 | 0.0320 | 0.0511 | yes |
| 5 | Trend Pullback | vol | fixed_atr | 7 | 12 | 10.0000 | 1.8248 | 199.4422 | 36.3352 | 0.0309 | 0.0218 | 0.0347 | yes |

Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? **YES**
Accepted candidates only (after min_holdout_trades / rejection filters): **NO**
