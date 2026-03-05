# Stage-34 Failure Modes

- wf_executed_pct: `33.33`
- mc_trigger_pct: `33.33`
- final_verdict: `NO_EDGE`

## Failure Mode Counts
- `no_trades_due_to_cost_drag`: `4`
- `no_trades_due_to_thresholds`: `8`

## Top Candidate Rows
| model | cost | window | trades | exp_lcb | pf_adj | maxdd_p95 | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hgbt | research | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| hgbt | live | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| hgbt | research | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| hgbt | live | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| rf | research | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| rf | live | 3 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| rf | research | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| rf | live | 6 | 0 | 0.000000 | 0.000000 | 0.000000 | no_trades_due_to_thresholds |
| logreg | research | 6 | 106667 | -4.522647 | 0.809107 | 0.999999 | no_trades_due_to_cost_drag |
| logreg | live | 6 | 106793 | -5.968350 | 0.729230 | 1.000000 | no_trades_due_to_cost_drag |
