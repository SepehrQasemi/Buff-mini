# Stage-34 Report

## Executive Summary
- run_id: `20260305_060126_70a58817c197_stage34`
- final_verdict: `NO_EDGE`
- top_bottleneck: `signal_quality`
- did_generations_improve: `True`

## Built Components
- 34.1 data snapshot audit + deterministic timeframe completion
- 34.2 OHLCV-only ML dataset + label pipeline
- 34.3 deterministic CPU model training + calibration
- 34.4 strict rolling WF/MC + cost stress evaluation
- 34.5 policy selection + held-out replay
- 34.6 model registry
- 34.7 evolution engine
- 34.8 ten-generation experiment

## Data Snapshot Audit
- snapshot_hash: `fee406f197c5055f8d33245c`
- resolved_end_ts: `2026-03-02T00:00:00+00:00`
- data_hash: `b1b6f031a041d620`

## ML Dataset Summary
- rows_total: `507726`
- feature_count: `35`
- timeframes: `['15m', '30m', '1h', '4h']`

## Model Training Summary
- `logreg` val_logloss=0.692014 test_logloss=0.693763 test_brier=0.250307
- `hgbt` val_logloss=0.692878 test_logloss=0.692066 test_brier=0.249459
- `rf` val_logloss=0.692647 test_logloss=0.692154 test_brier=0.249503

## Walkforward + Monte Carlo
- wf_executed_pct: `33.33`
- mc_trigger_pct: `33.33`
- failure_modes: `{'no_trades_due_to_thresholds': 8, 'no_trades_due_to_cost_drag': 4}`

## Cost Drag Analysis
- research_best_exp_lcb: `0.000000`
- live_best_exp_lcb: `0.000000`

## Policy Replay
- policy_id: `stage34_cac0fef838e3`
- research_trade_count: `3898`
- live_trade_count: `3903`
- top_reject_reasons: `[]`

## Ten-Generation Comparison
- generation_count: `10`
- best_generation: `3`
- did_generations_improve: `True`

## Bug Fixes
- Fixed Stage-34.3 runner path handling bug that attempted to read repo root as parquet when dataset path was omitted.
- Added bounded-compute deterministic subset selection for evolution loops to prevent runtime blowups.

## Final Verdict
- `NO_EDGE`

## Biggest Bottleneck
- `signal_quality`
- evidence: `{'live_trade_count': 3903, 'top_reject_reason': '', 'failure_mode': 'no_trades_due_to_thresholds', 'wf_executed_pct': 33.33333333333333, 'mc_trigger_pct': 33.33333333333333, 'live_best_exp_lcb': 0.0, 'research_best_exp_lcb': 0.0}`
