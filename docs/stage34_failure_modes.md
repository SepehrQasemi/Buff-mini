# Stage-34 Failure Modes

- run_id: `20260305_034621_9856bec4af07_stage34_eval`
- final_verdict: `NO_EDGE`
- wf_executed_pct: `33.33`
- mc_trigger_pct: `33.33`

## Failure Counts
- `no_trades_due_to_cost_drag`: `4`
- `no_trades_due_to_thresholds`: `8`

## Top Rows
| model | cost | window_m | trades | exp_lcb | pf_adj | maxdd_p95 | wf | mc | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| hgbt | research | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| hgbt | live | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| hgbt | research | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| hgbt | live | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| rf | research | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| rf | live | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| rf | research | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| rf | live | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | False | False | no_trades_due_to_thresholds |
| logreg | research | 6 | 106667 | -4.522647 | 0.809107 | 0.999999 | True | True | no_trades_due_to_cost_drag |
| logreg | live | 6 | 106793 | -5.968350 | 0.729230 | 1.000000 | True | True | no_trades_due_to_cost_drag |
