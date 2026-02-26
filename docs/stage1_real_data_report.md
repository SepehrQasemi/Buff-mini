# Stage-1 Real Data Report

- run_id: `20260226_192036_d518b36605f0_stage1`
- split_mode: `60_20_20`
- recent_weight: `2.0`
- min_holdout_trades: `50`
- promotion_holdout_months: `[3, 6, 9, 12]`
- promotion_counts: `{'6': 0, '9': 1, '12': 43}`
- min_validation_exposure_ratio: `0.01`
- min_validation_active_days: `10.0`
- validation_evidence_rule: `validation_exposure_ratio >= min_validation_exposure_ratio OR validation_active_days >= min_validation_active_days`
- acceptance_rule: `exp_lcb_holdout > 0 AND effective_edge > 0 AND validation_evidence_passed`
- rejected_due_validation_evidence_count: `37`
- target_trades_per_month_holdout: `8.0`
- low_signal_penalty_weight: `1.0`
- min_trades_per_month_floor: `2.0`
- round_trip_cost_pct: `0.1`

## Top 5 Candidates
| rank | strategy | gating | exit | holdout_m | val_trades | hold_trades | tpm_hold | pf_adj | exp_lcb | edge | exp_lcb>0 | edge>0 | accepted | exposure | PF_val | PF_hold | exp_val | exp_hold | max_dd_hold | return_hold | CAGR_hold | score |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Trend Pullback | none | fixed_atr | 3 | 47 | 28 | 9.1304 | 1.4936 | 30.5615 | 279.0398 | yes | yes | yes | 0.1073 | 1.1079 | 2.3750 | 11.8822 | 71.8697 | 0.0417 | 0.1006 | 0.4632 | 296.6814 |
| 2 | Trend Pullback | vol | fixed_atr | 3 | 11 | 8 | 2.6087 | 2.2414 | 58.7705 | 153.3145 | yes | yes | yes | 0.0251 | 1.6935 | 10.0000 | 47.8165 | 119.5705 | 0.0102 | 0.0478 | 0.2038 | 277.5148 |
| 3 | Trend Pullback | vol | fixed_atr | 3 | 18 | 8 | 2.6087 | 2.2414 | 39.3389 | 102.6232 | yes | yes | yes | 0.0204 | 1.8124 | 10.0000 | 60.4350 | 98.7015 | 0.0170 | 0.0395 | 0.1662 | 204.8794 |
| 4 | Trend Pullback | vol | breakeven_1r | 3 | 20 | 9 | 2.9348 | 2.3729 | 44.3964 | 130.2938 | yes | yes | yes | 0.0326 | 0.9695 | 10.0000 | 12.8437 | 87.3513 | 0.0032 | 0.0393 | 0.1654 | 144.3810 |
| 5 | Trend Pullback | vol | fixed_atr | 12 | 10 | 27 | 2.2192 | 1.2357 | 9.1897 | 20.3936 | yes | yes | yes | 0.0237 | 1.3338 | 1.6723 | 23.8623 | 54.9005 | 0.0554 | 0.0741 | 0.0742 | 19.8086 |

### Holdout By Symbol (Top Candidates)
- Rank 1 Trend Pullback: BTC(PF=1.4374, ret=0.0489, DD=0.0537, trades=15) | ETH(PF=3.3126, ret=0.1523, DD=0.0297, trades=13)
- Rank 2 Trend Pullback: BTC(PF=3.0297, ret=0.0452, DD=0.0200, trades=6) | ETH(PF=10.0000, ret=0.0504, DD=0.0005, trades=2)
- Rank 3 Trend Pullback: BTC(PF=1.7081, ret=0.0239, DD=0.0336, trades=6) | ETH(PF=10.0000, ret=0.0550, DD=0.0005, trades=2)
- Rank 4 Trend Pullback: BTC(PF=7.7227, ret=0.0561, DD=0.0050, trades=7) | ETH(PF=10.0000, ret=0.0225, DD=0.0015, trades=2)
- Rank 5 Trend Pullback: BTC(PF=1.3051, ret=0.0254, DD=0.0601, trades=14) | ETH(PF=2.0395, ret=0.1228, DD=0.0507, trades=13)

### Validation Evidence (OR Rule)
- Rank 1 Trend Pullback: exposure=0.0827 (pass=yes), active_days=72 (pass=yes), OR_passed=yes
- Rank 2 Trend Pullback: exposure=0.0157 (pass=yes), active_days=18 (pass=yes), OR_passed=yes
- Rank 3 Trend Pullback: exposure=0.0274 (pass=yes), active_days=26 (pass=yes), OR_passed=yes
- Rank 4 Trend Pullback: exposure=0.0259 (pass=yes), active_days=28 (pass=yes), OR_passed=yes
- Rank 5 Trend Pullback: exposure=0.0139 (pass=yes), active_days=13 (pass=yes), OR_passed=yes

## Accepted Summary
- Best ACCEPTED candidate: `Trend Pullback` (holdout_months_used=3, pf_adj_holdout=1.4936, exp_lcb_holdout=30.5615, effective_edge=279.0398, tpm_holdout=9.1304, exposure_ratio=0.1073, validation_exposure=0.0827, validation_active_days=72, score=296.6814)
- Top 10 accepted candidates tpm distribution: min=2.2192, median=2.6087, max=9.1304 (count=5)

Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? **YES**
Accepted candidates only (after validation evidence + exp_lcb/effective_edge + degeneracy filters): **YES**
