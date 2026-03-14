# Stage-28 Master Report

- run_id: `20260313_154858_044fff9053df_stage28`
- mode: `both`
- dry_run: `False`
- data_snapshot_id: `DATA_FROZEN_v1`
- data_snapshot_hash: `c734cebc1e80bf15`

## Window Counts
- 3m: generated=450, evaluated=450, expected=450
- 6m: generated=420, evaluated=420, expected=420

- wf_executed_pct: `100.000000`
- mc_trigger_pct: `100.000000`

## Policy Metrics
- research: `{'trade_count': 203.0, 'tpm': 0.5956184390689182, 'PF_raw': 0.46812670382289273, 'PF_clipped': 0.46812670382289273, 'expectancy': 28.58164916042464, 'exp_lcb': 0.6379702891539214, 'maxDD': 0.0421123952306923, 'zero_trade_pct': 71.42857142857143, 'invalid_pct': 0.0}`
- live: `{'trade_count': 202.0, 'tpm': 0.5926843580882831, 'PF_raw': 0.4483340156123745, 'PF_clipped': 0.4483340156123745, 'expectancy': 24.67872052777208, 'exp_lcb': -2.8708355847613793, 'maxDD': 0.05030922671316433, 'zero_trade_pct': 71.42857142857143, 'invalid_pct': 0.0}`

## Feasibility
- shadow_live_reject_rate: `0.128540`
- avg_feasible_pct_by_equity: `{'100': 100.0, '1000': 100.0, '10000': 100.0, '100000': 100.0}`

## Top Contextual Edges
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/4h | exp_lcb=786.308737 | trades=1 | occ=9
- MomentumBurst | VOLUME_SHOCK | BTC/USDT/4h | exp_lcb=786.131435 | trades=1 | occ=8
- MeanRevertAfterSpike | VOLUME_SHOCK | BTC/USDT/4h | exp_lcb=656.796216 | trades=1 | occ=8
- MomentumBurst | VOLUME_SHOCK | ETH/USDT/2h | exp_lcb=479.961827 | trades=1 | occ=22
- MeanRevertAfterSpike | VOLUME_SHOCK | ETH/USDT/2h | exp_lcb=428.532955 | trades=1 | occ=20
- MeanRevertAfterSpike | RANGE | BTC/USDT/4h | exp_lcb=341.886328 | trades=6 | occ=211
- MomentumBurst | VOLUME_SHOCK | BTC/USDT/2h | exp_lcb=340.883074 | trades=1 | occ=31
- MeanRevertAfterSpike | VOLUME_SHOCK | ETH/USDT/4h | exp_lcb=300.769199 | trades=2 | occ=17
- StructureBreak | TREND | BTC/USDT/30m | exp_lcb=-29.581749 | trades=71 | occ=1435
- VolExpansionContinuation | VOL_EXPANSION | BTC/USDT/1h | exp_lcb=-39.245277 | trades=113 | occ=633

## Verdict
- `WEAK_EDGE`
- next_bottleneck: `cost_drag_vs_signal`
