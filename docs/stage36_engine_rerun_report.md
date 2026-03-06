# Stage-36 Engine Rerun Report

Generated: 2026-03-06T03:05:30.446549+00:00

## Runs
- baseline_run_id: `20260306_011823_2f6281656c8b_stage28` (`configs/default.yaml`)
- enriched_run_id: `20260306_020928_226ede64fca7_stage28` (`configs/stage36_enriched_full.yaml`)

## Metric Comparison
- wf_executed_pct: baseline=100.0, enriched=100.0, delta=0.0
- mc_trigger_pct: baseline=100.0, enriched=100.0, delta=0.0
- research_trade_count: baseline=0.0, enriched=0.0, delta=0.0
- live_trade_count: baseline=0.0, enriched=0.0, delta=0.0
- research_best_exp_lcb: baseline=0.0, enriched=0.0, delta=0.0
- live_best_exp_lcb: baseline=0.0, enriched=0.0, delta=0.0

## Verdict
- baseline_verdict: `NO_EDGE`
- enriched_verdict: `NO_EDGE`
- promising: `False`
- policy_activated_baseline: `False`
- policy_activated_enriched: `False`

## Failure Modes
- baseline top_contextual_edges: `[{'candidate_id': '48c0ecdcac05', 'candidate': 'MomentumBurst', 'context': 'VOLUME_SHOCK', 'symbol': 'ETH/USDT', 'timeframe': '4h', 'exp_lcb': 786.3087371745651, 'trades_in_context': 1, 'context_occurrences': 9, 'classification': 'RARE'}, {'candidate_id': '2210ff9bfcf2', 'candidate': 'MomentumBurst', 'context': 'VOLUME_SHOCK', 'symbol': 'BTC/USDT', 'timeframe': '4h', 'exp_lcb': 786.1314348059955, 'trades_in_context': 1, 'context_occurrences': 8, 'classification': 'RARE'}, {'candidate_id': '99f846eada2e', 'candidate': 'MeanRevertAfterSpike', 'context': 'VOLUME_SHOCK', 'symbol': 'BTC/USDT', 'timeframe': '4h', 'exp_lcb': 656.7962159072638, 'trades_in_context': 1, 'context_occurrences': 8, 'classification': 'RARE'}, {'candidate_id': '38c4571bb1a9', 'candidate': 'VolExpansionContinuation', 'context': 'VOL_EXPANSION', 'symbol': 'BTC/USDT', 'timeframe': '1h', 'exp_lcb': -39.245276652257935, 'trades_in_context': 113, 'context_occurrences': 633, 'classification': 'FAIL'}, {'candidate_id': '2a1dd2a51046', 'candidate': 'FailedBreakReversal', 'context': 'RANGE', 'symbol': 'ETH/USDT', 'timeframe': '1h', 'exp_lcb': -47.58304256200516, 'trades_in_context': 63, 'context_occurrences': 1495, 'classification': 'FAIL'}, {'candidate_id': 'e9ea717f8856', 'candidate': 'MeanRevertAfterSpike', 'context': 'RANGE', 'symbol': 'ETH/USDT', 'timeframe': '4h', 'exp_lcb': -289.6631146771689, 'trades_in_context': 7, 'context_occurrences': 187, 'classification': 'FAIL'}]`
- enriched top_contextual_edges: `[{'candidate_id': '48c0ecdcac05', 'candidate': 'MomentumBurst', 'context': 'VOLUME_SHOCK', 'symbol': 'ETH/USDT', 'timeframe': '4h', 'exp_lcb': 786.3087371745651, 'trades_in_context': 1, 'context_occurrences': 9, 'classification': 'RARE'}, {'candidate_id': '2210ff9bfcf2', 'candidate': 'MomentumBurst', 'context': 'VOLUME_SHOCK', 'symbol': 'BTC/USDT', 'timeframe': '4h', 'exp_lcb': 786.1314348059955, 'trades_in_context': 1, 'context_occurrences': 8, 'classification': 'RARE'}, {'candidate_id': '99f846eada2e', 'candidate': 'MeanRevertAfterSpike', 'context': 'VOLUME_SHOCK', 'symbol': 'BTC/USDT', 'timeframe': '4h', 'exp_lcb': 656.7962159072638, 'trades_in_context': 1, 'context_occurrences': 8, 'classification': 'RARE'}, {'candidate_id': '38c4571bb1a9', 'candidate': 'VolExpansionContinuation', 'context': 'VOL_EXPANSION', 'symbol': 'BTC/USDT', 'timeframe': '1h', 'exp_lcb': -39.245276652257935, 'trades_in_context': 113, 'context_occurrences': 633, 'classification': 'FAIL'}, {'candidate_id': '2a1dd2a51046', 'candidate': 'FailedBreakReversal', 'context': 'RANGE', 'symbol': 'ETH/USDT', 'timeframe': '1h', 'exp_lcb': -47.58304256200516, 'trades_in_context': 63, 'context_occurrences': 1495, 'classification': 'FAIL'}, {'candidate_id': 'e9ea717f8856', 'candidate': 'MeanRevertAfterSpike', 'context': 'RANGE', 'symbol': 'ETH/USDT', 'timeframe': '4h', 'exp_lcb': -289.6631146771689, 'trades_in_context': 7, 'context_occurrences': 187, 'classification': 'FAIL'}]`
- execution_reject_breakdown_baseline: `{}`
- execution_reject_breakdown_enriched: `{}`

## Stability Check
- 5-seed stability run skipped because run is not promising under the defined conservative gate.
