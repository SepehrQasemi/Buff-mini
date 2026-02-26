# Stage-1 Real Data Report

- run_id: `20260226_181107_dd4955695aad_stage1`
- split_mode: `60_20_20`
- recent_weight: `2.0`
- min_holdout_trades: `50`
- min_validation_exposure_ratio: `0.01`
- min_validation_active_days: `10.0`
- rejected_due_validation_evidence_count: `17`
- target_trades_per_month_holdout: `8.0`
- low_signal_penalty_weight: `1.0`
- min_trades_per_month_floor: `2.0`
- round_trip_cost_pct: `0.1`

## Top 5 Candidates
| rank | strategy | gating | exit | val_trades | hold_trades | tpm_hold | pf_adj | exp_lcb | effective_edge | exposure | PF_val | PF_hold | exp_val | exp_hold | max_dd_hold | return_hold | CAGR_hold | score | rejected |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Trend Pullback | vol | fixed_atr | 10 | 19 | 2.4733 | 1.1234 | -12.1306 | -30.0029 | 0.0274 | 10.0000 | 1.4481 | 63.8135 | 39.5547 | 0.0582 | 0.0376 | 0.0602 | -60.4140 | no |
| 2 | Trend Pullback | vol+regime | fixed_atr | 0 | 2 | 0.2604 | 1.3462 | 282.5295 | 73.5568 | 0.0014 | 0.0000 | 10.0000 | 0.0000 | 298.2824 | 0.0005 | 0.0298 | 0.0477 | -1000000000.0000 | yes |
| 3 | Trend Pullback | vol | fixed_atr | 0 | 1 | 0.1302 | 1.1765 | 265.8505 | 34.6072 | 0.0004 | 0.0000 | 10.0000 | 0.0000 | 265.8505 | 0.0003 | 0.0133 | 0.0211 | -1000000000.0000 | yes |
| 4 | Trend Pullback | vol+regime | fixed_atr | 0 | 1 | 0.1302 | 1.1765 | 201.0210 | 26.1680 | 0.0002 | 0.0000 | 10.0000 | 0.0000 | 201.0210 | 0.0003 | 0.0101 | 0.0160 | -1000000000.0000 | yes |
| 5 | Trend Pullback | vol+regime | fixed_atr | 0 | 1 | 0.1302 | 1.1765 | 229.1560 | 29.8305 | 0.0003 | 0.0000 | 10.0000 | 0.0000 | 229.1560 | 0.0003 | 0.0115 | 0.0182 | -1000000000.0000 | yes |

### Holdout By Symbol (Top Candidates)
- Rank 1 Trend Pullback: BTC(PF=1.3774, ret=0.0292, DD=0.0535, trades=11) | ETH(PF=1.5188, ret=0.0459, DD=0.0629, trades=8)
- Rank 2 Trend Pullback: BTC(PF=10.0000, ret=0.0276, DD=0.0005, trades=1) | ETH(PF=10.0000, ret=0.0321, DD=0.0005, trades=1)
- Rank 3 Trend Pullback: BTC(PF=10.0000, ret=0.0266, DD=0.0005, trades=1) | ETH(PF=0.0000, ret=0.0000, DD=0.0000, trades=0)
- Rank 4 Trend Pullback: BTC(PF=10.0000, ret=0.0201, DD=0.0005, trades=1) | ETH(PF=0.0000, ret=0.0000, DD=0.0000, trades=0)
- Rank 5 Trend Pullback: BTC(PF=10.0000, ret=0.0229, DD=0.0005, trades=1) | ETH(PF=0.0000, ret=0.0000, DD=0.0000, trades=0)

## Accepted Summary
- Best ACCEPTED candidate: `Trend Pullback` (pf_adj_holdout=1.1234, exp_lcb_holdout=-12.1306, tpm_holdout=2.4733, exposure_ratio=0.0274, validation_exposure=0.0202, validation_active_days=9.29, score=-60.4140)
- Top 10 accepted candidates tpm distribution: min=2.4733, median=2.4733, max=2.4733 (count=1)

Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? **YES**
Accepted candidates only (after low-signal degeneracy filter): **YES**
