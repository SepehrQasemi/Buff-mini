# Stage-1 Real Data Report

- run_id: `20260226_170035_3fc0a2b58f96_stage1`
- split_mode: `60_20_20`
- recent_weight: `2.0`
- min_holdout_trades: `50`
- target_trades_per_month_holdout: `8.0`
- low_signal_penalty_weight: `1.0`
- min_trades_per_month_floor: `2.0`
- round_trip_cost_pct: `0.1`

## Top 5 Candidates
| rank | strategy | gating | exit | val_trades | hold_trades | tpm_hold | pf_adj | exp_lcb | effective_edge | exposure | PF_val | PF_hold | exp_val | exp_hold | max_dd_hold | return_hold | CAGR_hold | score | rejected |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Trend Pullback | vol+regime | fixed_atr | 16 | 12 | 1.5621 | 1.5885 | 59.9725 | 93.6833 | 0.0241 | 0.7914 | 4.0404 | -1.6364 | 133.1680 | 0.0239 | 0.0799 | 0.1296 | 177.0776 | no |
| 2 | Trend Pullback | vol+regime | fixed_atr | 0 | 2 | 0.2604 | 1.3462 | 282.5295 | 73.5568 | 0.0014 | 0.0000 | 10.0000 | 0.0000 | 298.2824 | 0.0005 | 0.0298 | 0.0477 | 122.6335 | no |
| 3 | Trend Pullback | vol+regime | fixed_atr | 2 | 6 | 0.7811 | 1.9643 | 87.1584 | 68.0753 | 0.0025 | 10.0000 | 10.0000 | 84.1132 | 106.5357 | 0.0005 | 0.0320 | 0.0511 | 101.8855 | no |
| 4 | Trend Pullback | vol+regime | fixed_atr | 2 | 6 | 0.7811 | 1.9643 | 66.0381 | 51.5792 | 0.0052 | 10.0000 | 10.0000 | 131.8402 | 127.9561 | 0.0076 | 0.0384 | 0.0615 | 85.0970 | no |
| 5 | Trend Pullback | vol | fixed_atr | 0 | 2 | 0.2604 | 1.3462 | 201.6304 | 52.4946 | 0.0013 | 0.0000 | 10.0000 | 0.0000 | 250.8928 | 0.0005 | 0.0251 | 0.0401 | 85.0332 | no |

## Accepted Summary
- Best ACCEPTED candidate: `Trend Pullback` (pf_adj_holdout=1.5885, exp_lcb_holdout=59.9725, tpm_holdout=1.5621, exposure_ratio=0.0241, score=177.0776)
- Top 10 accepted candidates tpm distribution: min=0.2604, median=0.7811, max=1.5621 (count=5)

Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? **YES**
Accepted candidates only (after low-signal degeneracy filter): **YES**
