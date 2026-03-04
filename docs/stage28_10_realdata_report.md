# Stage-28.10 Real-Data Validation Report

## 1) Data Status
- Symbols used: `['BTC/USDT']`
- Coverage years: `{'BTC/USDT': 4.0}`
- Snapshot: `DATA_FROZEN_v1` / `c734cebc1e80bf15`
- Timeframes: `['1h', '4h']`

## 2) Window Execution
- Step months: `1`
- Per-timeframe window counts: `{'1h': {'3m': 45, '6m': 42}, '4h': {'3m': 45, '6m': 42}}`
- Aggregate counts: `{'3m': 90, '6m': 84}`
- Note: counts are per timeframe; with two TFs, aggregate is doubled.

## 3) Funnel Stats
- StageA/B/C: `{'stage_a_count': 2628, 'stage_b_count': 12, 'stage_c_count': 12, 'stage_b_exploration_count': 4, 'stage_b_exploration_pct': 0.3333333333333333, 'sim_threshold': 0.9, 'exploration_pct_configured': 0.15}`
- Exploration quota used: `33.33%`

## 4) Validation Stats
- wf_executed_pct: `100.000000`
- mc_trigger_pct: `100.000000`
- WF-validated candidates: `12`
- MC-validated candidates: `12`
- Costs: `{'round_trip_cost_pct': 0.1, 'slippage_pct': 0.0005, 'cost_model_mode': 'simple'}`

### Top 5 Candidates by exp_lcb (with evidence sizes)
| candidate_id | candidate | context | timeframe | exp_lcb | trades | occurrences | windows_positive_ratio |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2210ff9bfcf2 | MomentumBurst | VOLUME_SHOCK | 4h | 786.131435 | 1 | 8 | 0.114943 |
| 99f846eada2e | MeanRevertAfterSpike | VOLUME_SHOCK | 4h | 656.796216 | 1 | 8 | 0.229885 |
| cad8cb799e32 | MeanRevertAfterSpike | RANGE | 4h | 341.886328 | 6 | 211 | 0.114943 |
| 00dade34e2fb | FailedBreakReversal | VOL_EXPANSION | 4h | 252.879879 | 1 | 68 | 0.011494 |
| e1629e9b3f6b | MomentumBurst | VOLUME_SHOCK | 1h | 67.777492 | 9 | 53 | 0.057471 |

## 5) Policy Replay
- Research metrics: `{'trade_count': 0.0, 'tpm': 0.0, 'PF_raw': 0.0, 'PF_clipped': 0.0, 'expectancy': 0.0, 'exp_lcb': 0.0, 'maxDD': 0.0, 'zero_trade_pct': 100.0, 'invalid_pct': 0.0}`
- Live metrics: `{'trade_count': 0.0, 'tpm': 0.0, 'PF_raw': 0.0, 'PF_clipped': 0.0, 'expectancy': 0.0, 'exp_lcb': 0.0, 'maxDD': 0.0, 'zero_trade_pct': 100.0, 'invalid_pct': 0.0}`
- Feasibility summary: `{'shadow_live_reject_rate': 0.0, 'shadow_live_top_reasons': {}, 'avg_feasible_pct_by_equity': {}}`

## 6) Final Verdict
- Verdict: `NO_EDGE`
- Biggest bottleneck: `policy_activation_thresholds`
- Evidence: finalists and WF/MC run, but `policy.json` has zero contexts selected (`no_rows_passed_policy_thresholds`), yielding zero replay trades.
