# Stage-1 Real Data Report

- run_id: `20260226_234139_b78b63b21ba0_stage1`
- split_mode: `60_20_20`
- recent_weight: `2.0`
- min_holdout_trades: `50`
- promotion_holdout_months: `[3, 6, 9, 12]`
- promotion_counts: `{'6': 0, '9': 1, '12': 43}`
- result_thresholds: `{'min_exp_lcb_holdout': 0, 'min_effective_edge': 0, 'min_trades_per_month_holdout': 5, 'min_pf_adj_holdout': 1.1, 'max_drawdown_holdout': 0.15, 'min_exposure_ratio': 0.02}`
- min_validation_exposure_ratio: `0.01`
- min_validation_active_days: `10.0`
- validation_evidence_rule: `validation_exposure_ratio >= min_validation_exposure_ratio OR validation_active_days >= min_validation_active_days`
- Tier A rule: `all result_thresholds satisfied`
- Tier B rule: `exp_lcb_holdout > 0 AND trades_per_month_holdout >= 3`
- Near Miss rule: `exp_lcb_holdout > -5`
- rejected_due_validation_evidence_count: `37`
- Tier A count: `1`
- Tier B count: `0`
- near_miss_count: `29`
- candidates_dir: `candidates`
- tier_A_csv: `tier_A_candidates.csv`
- tier_B_csv: `tier_B_candidates.csv`
- near_miss_csv: `near_miss_candidates.csv`
- stage_c_seconds: `551.50`
- target_trades_per_month_holdout: `8.0`
- low_signal_penalty_weight: `1.0`
- min_trades_per_month_floor: `2.0`
- round_trip_cost_pct: `0.1`

## Tier A Candidates
| rank | strategy | gating | exit | holdout_m | val_trades | hold_trades | tpm_hold | pf_adj | exp_lcb | edge | exposure | PF_val | PF_hold | exp_val | exp_hold | max_dd_hold | return_hold | CAGR_hold | score |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Trend Pullback | none | fixed_atr | 3 | 47 | 28 | 9.1304 | 1.4936 | 30.5615 | 279.0398 | 0.1073 | 1.1079 | 2.3750 | 11.8822 | 71.8697 | 0.0417 | 0.1006 | 0.4632 | 296.6814 |

### Holdout By Symbol (Tier A)
- Rank 1 Trend Pullback: BTC(PF=1.4374, ret=0.0489, DD=0.0537, trades=15) | ETH(PF=3.3126, ret=0.1523, DD=0.0297, trades=13)

### Validation Evidence (Tier A)
- Rank 1 Trend Pullback: exposure=0.0827 (pass=yes), active_days=72 (pass=yes), OR_passed=yes

## Tier Summary
- Best Tier A candidate: `Trend Pullback` (holdout_months_used=3, pf_adj_holdout=1.4936, exp_lcb_holdout=30.5615, effective_edge=279.0398, tpm_holdout=9.1304, exposure_ratio=0.1073, validation_exposure=0.0827, validation_active_days=72, score=296.6814)
- Tier A count: `1`
- Tier B count: `0`
- Near Miss count: `29`

Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? **YES**
Tier A candidates only: **YES**
