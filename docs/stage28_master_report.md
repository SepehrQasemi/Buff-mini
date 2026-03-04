# Stage-28 Master Report

- run_id: `20260304_190410_215b4ee71cfc_stage28`
- mode: `live`
- dry_run: `True`
- data_snapshot_id: `DATA_FROZEN_v1`
- data_snapshot_hash: `c734cebc1e80bf15`

## Window Counts
- 3m: generated=10, evaluated=10, expected=10
- 6m: generated=0, evaluated=0, expected=0

- wf_executed_pct: `0.000000`
- mc_trigger_pct: `0.000000`

## Policy Metrics
- research: `{'trade_count': 0.0, 'tpm': 0.0, 'PF_raw': 0.0, 'PF_clipped': 0.0, 'expectancy': 0.0, 'exp_lcb': 0.0, 'maxDD': 0.0, 'zero_trade_pct': 100.0, 'invalid_pct': 0.0}`
- live: `{'trade_count': 0.0, 'tpm': 0.0, 'PF_raw': 0.0, 'PF_clipped': 0.0, 'expectancy': 0.0, 'exp_lcb': 0.0, 'maxDD': 0.0, 'zero_trade_pct': 100.0, 'invalid_pct': 0.0}`

## Feasibility
- shadow_live_reject_rate: `0.000000`
- avg_feasible_pct_by_equity: `{}`

## Top Contextual Edges
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/1h | exp_lcb=148.429208 | trades=2 | occ=52
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/4h | exp_lcb=148.429208 | trades=2 | occ=52
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/30m | exp_lcb=148.429208 | trades=2 | occ=52
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/15m | exp_lcb=148.429208 | trades=2 | occ=52
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/2h | exp_lcb=148.429208 | trades=2 | occ=52
- TrendFlip | TREND | ETH/USDT/30m | exp_lcb=0.000000 | trades=0 | occ=830
- TrendFlip | TREND | ETH/USDT/15m | exp_lcb=0.000000 | trades=0 | occ=830
- TrendFlip | TREND | ETH/USDT/1h | exp_lcb=0.000000 | trades=0 | occ=830

## Verdict
- `NO_EDGE`
- next_bottleneck: `wf_usability`
