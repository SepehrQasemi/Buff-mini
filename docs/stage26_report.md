# Stage-26 Report

- head_commit: `35ba00ceec5ad478816d362170310724804bc570`
- run_id: `20260303_003944_de1d0180cfb4_stage26`
- seed: `42`
- dry_run: `False`

## Data Coverage (4y)
- coverage_ok_all_symbols: `False`
- required_years: `4.0`

## Metrics
| mode | trade_count | tpm | exp_lcb | maxDD | wf_executed_pct | mc_trigger_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional_research | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.00 | 0.00 |
| conditional_live | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.00 | 0.00 |
| global_baseline | 5.00 | 3.6856 | 216.922299 | 0.000250 | 0.00 | 0.00 |

## Conditional vs Global Delta
- delta_exp_lcb: `-216.922299`
- delta_maxDD: `-0.000250`
- delta_trade_count: `-5.000000`

## Shadow Live
- shadow_live_reject_rate: `0.000000`
- shadow_live_top_reasons: `{}`

## Verdict
- `INSUFFICIENT_DATA`
- next_bottleneck: `signal_quality`
