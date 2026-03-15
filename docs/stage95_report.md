# Stage-95 Report

- status: `SUCCESS`
- validation_state: `LIVE_USEFULNESS_READY`
- symbol: `BTC/USDT`
- timeframe: `1h`
- before_profile: `stage94_baseline`
- after_profile: `stage95_usefulness_push`
- stage95b_recommended: `True`
- stage95b_applied: `True`
- replay_window_bars: `2048`

## Before Counts
- candidate_count: `24`
- hierarchy_counts: `{'interesting_but_fragile': 18, 'junk': 6}`
- mean_near_miss_distance: `0.446763`
- mean_rank_score: `0.455476`
- near_miss_count: `21`
- promising_count: `0`
- replay_death_fraction: `0.375`
- useful_candidate_count: `18`

## After Counts
- candidate_count: `24`
- hierarchy_counts: `{'interesting_but_fragile': 18, 'junk': 6}`
- mean_near_miss_distance: `0.446763`
- mean_rank_score: `0.507069`
- near_miss_count: `21`
- promising_count: `0`
- replay_death_fraction: `0.375`
- useful_candidate_count: `18`

## Usefulness Delta
- mean_near_miss_delta: `0.0`
- mean_rank_score_delta: `0.051593`
- promising_delta: `0`
- replay_death_fraction_delta: `0.0`
- useful_candidate_delta: `0`

## Family Usefulness
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 0.0, "after_useful_candidate_count": 3, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 0.0, "before_useful_candidate_count": 3, "family": "structure_pullback_continuation", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 0.0, "after_useful_candidate_count": 3, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 0.0, "before_useful_candidate_count": 3, "family": "liquidity_sweep_reversal", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 1.0, "after_useful_candidate_count": 3, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 1.0, "before_useful_candidate_count": 3, "family": "squeeze_flow_breakout", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 0.0, "after_useful_candidate_count": 3, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 0.0, "before_useful_candidate_count": 3, "family": "failed_breakout_reversal", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 0.0, "after_useful_candidate_count": 3, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 0.0, "before_useful_candidate_count": 3, "family": "exhaustion_mean_reversion", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 0, "after_promising_count": 0, "after_replay_death_fraction": 1.0, "after_useful_candidate_count": 0, "before_candidate_count": 3, "before_near_miss_count": 0, "before_promising_count": 0, "before_replay_death_fraction": 1.0, "before_useful_candidate_count": 0, "family": "funding_oi_imbalance_reversion", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 0.0, "after_useful_candidate_count": 3, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 0.0, "before_useful_candidate_count": 3, "family": "volatility_regime_transition", "promising_delta": 0, "useful_delta": 0}`
- `{"after_candidate_count": 3, "after_near_miss_count": 3, "after_promising_count": 0, "after_replay_death_fraction": 1.0, "after_useful_candidate_count": 0, "before_candidate_count": 3, "before_near_miss_count": 3, "before_promising_count": 0, "before_replay_death_fraction": 1.0, "before_useful_candidate_count": 0, "family": "multi_tf_disagreement_repair", "promising_delta": 0, "useful_delta": 0}`

## Family Replay Death Map
- `{"candidate_count": 3, "family": "exhaustion_mean_reversion", "replay_death_fraction": 0.0, "replay_deaths": 0, "survivors": 3, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "failed_breakout_reversal", "replay_death_fraction": 0.0, "replay_deaths": 0, "survivors": 3, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "funding_oi_imbalance_reversion", "replay_death_fraction": 1.0, "replay_deaths": 3, "survivors": 0, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "liquidity_sweep_reversal", "replay_death_fraction": 0.0, "replay_deaths": 0, "survivors": 3, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "multi_tf_disagreement_repair", "replay_death_fraction": 1.0, "replay_deaths": 3, "survivors": 0, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "squeeze_flow_breakout", "replay_death_fraction": 1.0, "replay_deaths": 3, "survivors": 0, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "structure_pullback_continuation", "replay_death_fraction": 0.0, "replay_deaths": 0, "survivors": 3, "transfer_deaths": 0, "walkforward_deaths": 0}`
- `{"candidate_count": 3, "family": "volatility_regime_transition", "replay_death_fraction": 0.0, "replay_deaths": 0, "survivors": 3, "transfer_deaths": 0, "walkforward_deaths": 0}`

## Dead Weight Families
- `{"after_candidate_count": 3, "after_replay_death_fraction": 1.0, "family": "funding_oi_imbalance_reversion", "reason": "no_useful_survivors_and_replay_dominated"}`

- summary_hash: `64b3919bad947f8b`
