# Stage-99 Report

- status: `SUCCESS`
- validation_state: `CANDIDATE_QUALITY_ACCELERATION_READY`
- symbol: `BTC/USDT`
- timeframe: `1h`
- stage99b_required: `True`
- stage99b_applied: `True`

## Before Counts
- candidate_count: `400`
- interesting_count: `0`
- promising_count: `6`
- rejected_count: `0`
- robust_count: `0`
- validated_count: `0`

## After Counts
- candidate_count: `400`
- interesting_count: `0`
- promising_count: `6`
- rejected_count: `0`
- robust_count: `0`
- validated_count: `0`

## Transition Rows
- `{"after_final_class": "absent", "after_rank_score": 0.47407608984375005, "before_final_class": "promising_but_unproven", "before_rank_score": 0.51382163984375, "candidate_id": "s70_04ff8d2487b9cdc1", "change_reason": ["useful_trade_density_or_transfer_prior", "acceptable_trade_density", "thicker_evidence_profile"], "family": "structure_pullback_continuation", "rank_delta": -0.039746}`
- `{"after_final_class": "promising_but_unproven", "after_rank_score": 0.48484833984375003, "before_final_class": "promising_but_unproven", "before_rank_score": 0.51414833984375, "candidate_id": "s70_13a54f076f03d108", "change_reason": ["useful_trade_density_or_transfer_prior", "lower_clustering_risk"], "family": "structure_pullback_continuation", "rank_delta": -0.0293}`
- `{"after_final_class": "promising_but_unproven", "after_rank_score": 0.48519833984375, "before_final_class": "promising_but_unproven", "before_rank_score": 0.51379833984375, "candidate_id": "s70_1f4f067819f2535d", "change_reason": ["strong_trade_quality_bonus", "useful_trade_density_or_transfer_prior", "lower_clustering_risk"], "family": "liquidity_sweep_reversal", "rank_delta": -0.0286}`
- `{"after_final_class": "promising_but_unproven", "after_rank_score": 0.48484833984375003, "before_final_class": "promising_but_unproven", "before_rank_score": 0.51414833984375, "candidate_id": "s70_22e318cecdbfc696", "change_reason": ["useful_trade_density_or_transfer_prior", "lower_clustering_risk"], "family": "structure_pullback_continuation", "rank_delta": -0.0293}`
- `{"after_final_class": "promising_but_unproven", "after_rank_score": 0.48519833984375, "before_final_class": "promising_but_unproven", "before_rank_score": 0.51379833984375, "candidate_id": "s70_36fd31b49fe76519", "change_reason": ["strong_trade_quality_bonus", "useful_trade_density_or_transfer_prior", "lower_clustering_risk"], "family": "liquidity_sweep_reversal", "rank_delta": -0.0286}`
- `{"after_final_class": "promising_but_unproven", "after_rank_score": 0.48684833984375003, "before_final_class": "promising_but_unproven", "before_rank_score": 0.5163983398437499, "candidate_id": "s70_98f70c07d2a05a58", "change_reason": ["useful_trade_density_or_transfer_prior", "lower_clustering_risk"], "family": "structure_pullback_continuation", "rank_delta": -0.02955}`
- `{"after_final_class": "promising_but_unproven", "after_rank_score": 0.48519833984375, "before_final_class": "absent", "before_rank_score": 0.51379833984375, "candidate_id": "s70_fbd94d0d24f8a180", "change_reason": ["strong_trade_quality_bonus", "useful_trade_density_or_transfer_prior", "lower_clustering_risk"], "family": "liquidity_sweep_reversal", "rank_delta": -0.0286}`

## Gate Heatmap
- `{"count": 3, "family": "liquidity_sweep_reversal", "gate": "replay"}`
- `{"count": 3, "family": "structure_pullback_continuation", "gate": "replay"}`

## Near Miss Inventory
- none

## Top-K Truth Review
- `{"candidate_id": "s70_98f70c07d2a05a58", "family": "structure_pullback_continuation", "mc_issue": "replay_gate_failed", "replay_issue": "exp_lcb", "rescue_hint": "tighten_participation_and_transfer_exposure", "target_regime": "trend", "transfer_issue": "regime_mismatch,timing_instability", "why_it_surfaced": ["useful_trade_density_or_transfer_prior", "lower_clustering_risk"]}`
- `{"candidate_id": "s70_1f4f067819f2535d", "family": "liquidity_sweep_reversal", "mc_issue": "replay_gate_failed", "replay_issue": "exp_lcb", "rescue_hint": "tighten_participation_and_transfer_exposure", "target_regime": "range", "transfer_issue": "regime_mismatch,timing_instability", "why_it_surfaced": ["strong_trade_quality_bonus", "useful_trade_density_or_transfer_prior", "lower_clustering_risk"]}`
- `{"candidate_id": "s70_36fd31b49fe76519", "family": "liquidity_sweep_reversal", "mc_issue": "replay_gate_failed", "replay_issue": "exp_lcb", "rescue_hint": "tighten_participation_and_transfer_exposure", "target_regime": "transition", "transfer_issue": "regime_mismatch,timing_instability", "why_it_surfaced": ["strong_trade_quality_bonus", "useful_trade_density_or_transfer_prior", "lower_clustering_risk"]}`
- `{"candidate_id": "s70_fbd94d0d24f8a180", "family": "liquidity_sweep_reversal", "mc_issue": "replay_gate_failed", "replay_issue": "exp_lcb", "rescue_hint": "tighten_participation_and_transfer_exposure", "target_regime": "range", "transfer_issue": "regime_mismatch,timing_instability", "why_it_surfaced": ["strong_trade_quality_bonus", "useful_trade_density_or_transfer_prior", "lower_clustering_risk"]}`
- `{"candidate_id": "s70_13a54f076f03d108", "family": "structure_pullback_continuation", "mc_issue": "replay_gate_failed", "replay_issue": "exp_lcb", "rescue_hint": "tighten_participation_and_transfer_exposure", "target_regime": "trend", "transfer_issue": "regime_mismatch,timing_instability", "why_it_surfaced": ["useful_trade_density_or_transfer_prior", "lower_clustering_risk"]}`
- `{"candidate_id": "s70_22e318cecdbfc696", "family": "structure_pullback_continuation", "mc_issue": "replay_gate_failed", "replay_issue": "exp_lcb", "rescue_hint": "tighten_participation_and_transfer_exposure", "target_regime": "transition", "transfer_issue": "regime_mismatch,timing_instability", "why_it_surfaced": ["useful_trade_density_or_transfer_prior", "lower_clustering_risk"]}`

- summary_hash: `6d7c3ef6cec06fd5`
