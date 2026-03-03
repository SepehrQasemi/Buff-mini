# Stage-26 Report

- head_commit: `d639937d3ce4f8e6e2b5ae75712d8471b6ad1b9c`
- run_id: `20260303_023207_d87597c6f689_stage26`
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
| global_baseline | 2.00 | 2.9694 | 274.647802 | 0.000200 | 0.00 | 0.00 |

## Conditional vs Global Delta
- delta_exp_lcb: `-274.647802`
- delta_maxDD: `-0.000200`
- delta_trade_count: `-2.000000`

## Shadow Live
- shadow_live_reject_rate: `0.000000`
- shadow_live_top_reasons: `{}`

## Verdict
- `INSUFFICIENT_DATA`
- next_bottleneck: `signal_quality`
