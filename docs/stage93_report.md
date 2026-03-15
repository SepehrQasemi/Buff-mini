# Stage-93 Report

- status: `SUCCESS`
- success_inventory_count: `12`
- learning_trace_hash: `899ae4a2aa8a7c8e`

## Failure Taxonomy
- `{"count": 8, "failure": "low_trade_count"}`
- `{"count": 0, "failure": "cost_fragile"}`
- `{"count": 8, "failure": "transfer_fail"}`
- `{"count": 1, "failure": "walkforward_fail"}`
- `{"count": 0, "failure": "perturbation_fail"}`
- `{"count": 5, "failure": "clustering_fail"}`
- `{"count": 0, "failure": "regime_overfit"}`
- `{"count": 7, "failure": "evidence_thin"}`

## Adaptation Steps
- `{"action": "broaden_trade_density_constraints", "magnitude": 0.16, "reason": "low_trade_count"}`
- `{"action": "increase_similarity_penalty", "magnitude": 0.05, "reason": "clustering_fail"}`
- `{"action": "downweight_transfer_fragile_families", "magnitude": 0.64, "reason": "transfer_fail"}`
- `{"action": "favor_forward_stable_regimes", "magnitude": 0.1, "reason": "walkforward_fail"}`
- `{"action": "success_weighted_mechanism_retention", "magnitude": 1.0, "reason": "promising_inventory_present"}`

## Search Feedback
- `{"family_priority_adjustments": {"liquidity_sweep_reversal": 0.137304, "structure_pullback_continuation": 0.090619}, "feedback_hash": "b8e58ffe4fe82463", "threshold_guidance": ["broaden_trade_density", "diversify_mechanism_mix", "favor_lower_transfer_risk"]}`

- summary_hash: `5668ef9867fd0590`
