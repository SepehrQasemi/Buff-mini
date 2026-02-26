# Stage-1 Real Data Report

- run_id: `20260226_141335_3fc0a2b58f96_stage1`
- split_mode: `60_20_20`
- recent_weight: `2.0`
- min_holdout_trades: `50`
- target_trades_per_month_holdout: `8.0`
- low_signal_penalty_weight: `1.0`
- min_trades_per_month_floor: `2.0`
- round_trip_cost_pct: `0.1`

## Top 5 Candidates
| rank | strategy | gating | exit | val_trades | hold_trades | tpm_hold | penalty | relief | PF_val | PF_hold | exp_val | exp_hold | max_dd_combined | return_hold | CAGR_hold | rejected |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Trend Pullback | vol | fixed_atr | 0 | 2 | 0.2604 | 0.4837 | yes | 0.0000 | 10.0000 | 0.0000 | 250.8928 | 0.0005 | 0.0251 | 0.0401 | no |
| 2 | Trend Pullback | vol+regime | fixed_atr | 0 | 2 | 0.2604 | 0.4837 | yes | 0.0000 | 10.0000 | 0.0000 | 298.2824 | 0.0005 | 0.0298 | 0.0477 | no |
| 3 | Trend Pullback | vol+regime | fixed_atr | 2 | 6 | 0.7811 | 0.4512 | yes | 10.0000 | 10.0000 | 65.9201 | 166.4198 | 0.0076 | 0.0384 | 0.0615 | no |
| 4 | Trend Pullback | vol | fixed_atr | 0 | 4 | 0.5207 | 0.4675 | yes | 0.0000 | 10.0000 | 0.0000 | 390.2880 | 0.0063 | 0.0511 | 0.0823 | no |
| 5 | Trend Pullback | vol+regime | fixed_atr | 16 | 12 | 1.5621 | 0.4024 | yes | 0.7914 | 4.0404 | -38.4809 | 176.5109 | 0.0631 | 0.0799 | 0.1296 | no |

## Accepted Summary
- Best ACCEPTED candidate: `Trend Pullback` (PF_holdout=10.0000, expectancy_holdout=250.8928, tpm_holdout=0.2604, penalty=0.4837)
- Top 10 accepted candidates tpm distribution: min=0.2604, median=0.5207, max=1.5621 (count=5)

Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? **YES**
Accepted candidates only (after low-signal degeneracy filter): **YES**
