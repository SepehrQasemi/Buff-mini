# Stage-25 Master Report

## Scope
- Stage-25A: margin/caps correctness and research/live constraint split.
- Stage-25B: family quality in research mode and live feasibility replay.
- Stage-25.5: minimal exit/regime conditional levers with strict validation preserved.

## Run IDs
- research_run_id: `20260302_214558_e829535e1dda_stage25B`
- live_run_id: `20260302_214623_a3027d37b4f7_stage25B`
- regime_run_id: `20260302_214648_137cddfd6458_stage15_9_trace`

## Research vs Live
- research status: `NO_EDGE_IN_RESEARCH`
- live status: `NO_EDGE_IN_LIVE`
- research exp_lcb_best: `0.000000`
- live exp_lcb_best: `0.000000`

## Live Feasibility Replay
- promising_research_candidates_count: `10`
- survived_count: `0` / `10`
- survived_pct: `0.000000`
- live_replay_exp_lcb_median: `0.000000`

## Minimal Improvement Levers
- Exit upgrade selection:
  - rows: `80`
  - trailing_selected: `30`
  - fixed_selected: `50`
  - upgraded_count: `30`
- Regime-conditional activation deltas vs live baseline:
  - exp_lcb_best: `0.000000`
  - exp_lcb_median: `-2.192874`
  - trade_count_total: `-840.000000`
  - zero_trade_pct: `0.000000`
  - walkforward_executed_true_pct: `0.000000`
  - mc_trigger_rate: `0.000000`

## Bottleneck
- next_bottleneck: `{'type': 'WEAKEST_FAMILY', 'value': 'flow', 'exp_lcb_best': -44.165807153388926, 'zero_trade_pct': 0.0}`

## Final Verdict
- `NO_EDGE`
