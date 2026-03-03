# Stage-27 Master Report

- generated_at: `20260303_144119`
- head_commit: `ad8b0cd861b709fe49c18ea9bc74cadd82a7aa76`

## Data Status
- coverage_years_per_symbol: `{'BTC/USDT': 4.0, 'ETH/USDT': 4.0}`
- used_symbols: `['BTC/USDT', 'ETH/USDT']`
- data_snapshot_id: `DATA_FROZEN_v1`
- data_snapshot_hash: `c734cebc1e80bf15`

## Execution Health
- death_execution_rate: `0.672233`
- top_reject_reasons: `[{'reason': 'SIZE_TOO_SMALL', 'count': 21665}, {'reason': 'VALID', 'count': 615}]`
- feasibility_min_required_risk_floor: `{'15m': 9.48998922244559e-05, '1h': 9.48998922244559e-05, '2h': 9.48998922244559e-05, '30m': 9.48998922244559e-05, '4h': 9.48998922244559e-05}`

## Research Results
- global_baseline_metrics: `{'trade_count': 0.0, 'tpm': 0.0, 'PF_clipped': 0.0, 'expectancy': 0.0, 'exp_lcb': 0.0, 'maxDD': 0.0, 'zero_trade_pct': 100.0, 'invalid_pct': 0.0, 'walkforward_executed_true_pct': 0.0, 'usable_windows_count': 0, 'mc_trigger_rate': 0.0}`
- conditional_policy_metrics_live: `{'trade_count': 2020.0, 'tpm': 60.62526052521885, 'PF_clipped': 0.799062624790068, 'expectancy': -17.862442896140962, 'exp_lcb': -37.81852522197674, 'maxDD': 0.4432708142926822, 'zero_trade_pct': 0.0, 'invalid_pct': 0.0, 'walkforward_executed_true_pct': 0.0, 'usable_windows_count': 0, 'mc_trigger_rate': 0.0}`
- stage24_verdict: `SIZING_ACTIVE_NO_EDGE_CHANGE`
- stage25_research_verdict: `NO_EDGE_IN_RESEARCH`
- stage25_live_verdict: `NO_EDGE_IN_LIVE`
- stage26_verdict: `NO_EDGE`
- contextual_edge_rows: `10`
- contextual_policy_verdict: `DOES_POLICY_HAVE_CONTEXTUAL_EDGE`

## Final Verdict
- `CONTEXTUAL_EDGE_ONLY`
- next_bottleneck: `execution_feasibility:SIZE_TOO_SMALL`

## Next Actions
- If ETH coverage drops again, rerun canonical downloader and keep BTC-only fallback explicit.
- If SIZE_TOO_SMALL or POLICY_CAP_HIT dominates, apply feasibility-driven risk floor per timeframe/equity tier.
- If only contextual edge exists, deploy context-gated policy rather than global always-on policy.
