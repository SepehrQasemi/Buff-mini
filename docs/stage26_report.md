# Stage-26 Report

- head_commit: `ebde309639a2c39506fa54846470d1badc28bf75`
- run_id: `20260304_120151_f7d8fa27e646_stage26`
- seed: `42`
- dry_run: `True`

## Data Coverage (4y)
- coverage_ok_all_symbols: `True`
- required_years: `4.0`

## Metrics
| mode | trade_count | tpm | exp_lcb | maxDD | wf_executed_pct | mc_trigger_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_research | 2020.00 | 60.6253 | -36.840624 | 0.443271 | 0.00 | 0.00 |
| conditional_live | 2020.00 | 60.6253 | -37.818525 | 0.443271 | 0.00 | 0.00 |
| global_baseline | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.00 | 0.00 |

## Conditional vs Global Delta
- delta_exp_lcb: `-37.818525`
- delta_maxDD: `0.443271`
- delta_trade_count: `2020.000000`

## Shadow Live
- shadow_live_reject_rate: `0.972397`
- shadow_live_top_reasons: `{'SIZE_TOO_SMALL': 21665, 'VALID': 615}`

## Verdict
- `NO_EDGE`
- next_bottleneck: `live_feasibility_constraints`
