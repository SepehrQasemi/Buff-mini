# Stage-43 Performance Report

## Baseline vs Upgraded
- baseline_run_id: `20260306_160459_2a029423a621_stage28`
- upgraded_run_id: `20260306_144056_4b655c81c2a5_stage28`
- budget_mode: `small`

| metric | baseline | upgraded | delta |
| --- | ---: | ---: | ---: |
| raw_signal_count | 0.000000 | 0.000000 | 0.000000 |
| activation_rate | 0.000000 | 0.000000 | 0.000000 |
| trade_count | 0.000000 | 0.000000 | 0.000000 |
| research_best_exp_lcb | 0.000000 | 0.000000 | 0.000000 |
| live_best_exp_lcb | 0.000000 | 0.000000 | 0.000000 |
| wf_executed_pct | 100.000000 | 100.000000 | 0.000000 |
| mc_trigger_pct | 100.000000 | 100.000000 | 0.000000 |
| runtime_seconds | 2274.706470 | 2361.551806 | 86.845335 |

## Runtime By Phase (seconds)
| phase | seconds |
| --- | ---: |
| config_load | 0.050476 |
| data_load | 0.058270 |
| extras_alignment | 0.000008 |
| feature_generation | 0.003722 |
| candidate_generation | 0.011475 |
| stage_a_objective | 0.002303 |
| stage_b_objective | 0.000618 |
| composer_policy_build | 0.094816 |
| replay_backtest | 2361.551806 |
| walkforward | 0.000000 |
| monte_carlo | 0.000000 |
| report_generation | 0.002118 |

- slowest_phase: `replay_backtest`
- promising: `False`
- note: `run_stage28 exposes total runtime only; walkforward/monte_carlo remain embedded in replay_backtest runtime.`
