# Stage-40 Tradability Objective Report

## Objective Shift
- Stage-A objective targets activation + tradability before robustness.
- Stage-B objective keeps strict robustness filtering.

## Candidate Survival
- input_candidates: `5`
- stage_a_survivors: `3`
- stage_b_survivors: `3`
- before_strict_direct_survivors: `4`

## Label Stats
- tradable_rate: `0.178333`
- tp_before_sl_rate: `0.368333`
- net_return_after_cost_mean: `-0.001481`

## Bottleneck
- strongest_bottleneck_step: `stage_a_activation`

## Before vs After
- before (strict-direct): `4`
- after (two-stage, final): `3`
