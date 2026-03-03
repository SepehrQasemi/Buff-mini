# Stage-26 Report

- head_commit: `9d9c935d6636876366a3cbe68ff43e08e2c15a0e`
- run_id: `20260303_050348_e15ab06fe59c_stage26`
- seed: `42`
- dry_run: `False`

## Data Coverage (4y)
- coverage_ok_all_symbols: `False`
- required_years: `4.0`

## Metrics
| mode | trade_count | tpm | exp_lcb | maxDD | wf_executed_pct | mc_trigger_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_research | 783.00 | 11.7861 | 1.647819 | 0.477785 | 0.00 | 0.00 |
| conditional_live | 783.00 | 11.7861 | 2.573864 | 0.477785 | 0.00 | 0.00 |
| global_baseline | 1.00 | 3.7500 | 124.808451 | 0.000250 | 50.00 | 0.00 |

## Conditional vs Global Delta
- delta_exp_lcb: `-122.234588`
- delta_maxDD: `0.477535`
- delta_trade_count: `782.000000`

## Shadow Live
- shadow_live_reject_rate: `0.977959`
- shadow_live_top_reasons: `{'SIZE_TOO_SMALL': 8297, 'VALID': 187}`

## Verdict
- `INSUFFICIENT_DATA`
- next_bottleneck: `live_feasibility_constraints`
