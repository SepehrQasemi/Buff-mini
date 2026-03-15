# Stage-85 Report

- status: `SUCCESS`
- execution_status: `EXECUTED`
- stage_role: `real_validation`
- validation_state: `REALITY_MATRIX_READY`

## Reality Matrix
- synthetic_clean_easy: candidate_count=`9` promising_count=`6` validated_count=`0` robust_count=`0` blocked_count=`0`
- synthetic_clean_hard: candidate_count=`9` promising_count=`4` validated_count=`0` robust_count=`0` blocked_count=`0`
- live_relaxed: candidate_count=`400` promising_count=`0` validated_count=`0` robust_count=`0` blocked_count=`0`
- live_strict: candidate_count=`0` promising_count=`0` validated_count=`0` robust_count=`0` blocked_count=`1`

## Dominant Blockers
- data_canonicalization: `1`
- generator_depth: `8`
- ranking_funnel_pressure: `8`

## Gate Sensitivity
- `{"baseline_min_exp_lcb": 0.0, "baseline_min_trade_count": 40, "gate": "replay", "looser_trade_count": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 1, "synthetic_clean_hard": 1}, "stricter_trade_count": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}, "survivor_counts": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}}`
- `{"baseline_min_usable_windows": 5, "gate": "walkforward", "looser": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}, "stricter": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}, "survivor_counts": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}}`
- `{"gate": "monte_carlo", "survivor_counts": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 9, "synthetic_clean_hard": 9}}`
- `{"gate": "transfer", "survivor_counts": {"live_relaxed": 0, "live_strict": 0, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}}`
- `{"blocked_counts": {"live_relaxed": 0, "live_strict": 1, "synthetic_clean_easy": 0, "synthetic_clean_hard": 0}, "gate": "continuity"}`

- summary_hash: `fbc24b491b803c2c`
