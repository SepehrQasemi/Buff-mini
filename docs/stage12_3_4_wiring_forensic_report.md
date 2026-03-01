# Stage-12.3/12.4 Wiring Forensic Report

## Scope
- Objective: prove whether Stage-12.3/12.4 were no-op, identify root cause, fix wiring/ordering, and re-run with evidence.
- Runner entrypoint: `scripts/run_stage12.py`
- Flags:
  - Stage-12.3: `--stage12-3-enabled`
  - Stage-12.4: `--stage12-3-enabled --stage12-4-enabled`

## Reproduction Evidence (Before Fix)
Reference runs:
- Stage-12.3 pre-fix: `20260301_050945_72afe9b75d79_stage12`
- Stage-12.4 pre-fix: `20260301_051916_7adfeefe759b_stage12`

Observed metrics from `runs/<run_id>/stage12_forensic_summary.json`:

| mode | zero_trade_pct | walkforward_executed_true_pct | MC_trigger_rate | invalid_pct |
| --- | ---: | ---: | ---: | ---: |
| pre-fix Stage-12.3 | 62.878788 | 0.000000 | 0.000000 | 100.000000 |
| pre-fix Stage-12.4 | 71.969697 | 0.000000 | 0.000000 | 100.000000 |

Hard evidence for root cause:
1. `walkforward_expected_windows` was `0` for all combinations (no walkforward windows generated).
2. `stage12_trace.json` did not exist pre-fix, so no proof hooks executed.
3. Stage-12.4 scoring wrapper was not applied in walkforward holdout/forward path (evaluation used raw signals there).

## Root Cause (Proven)
1. **Early rejection before meaningful walkforward**:
   - With short local history, default Stage-8 window settings produced zero windows.
   - Result: `walkforward_executed_true_pct=0`, all non-zero-trade combos collapsed to `LOW_USABLE_WINDOWS`.
2. **Adaptive usability mismatch**:
   - Adaptive min-trades was computed with configured `forward_days=30` even when fallback forward windows were much shorter.
   - This inflated required trades per window and rejected viable combos.
3. **Stage-12.4 partial wiring**:
   - Scoring wrapper executed on full backtest path, but not in walkforward holdout/forward generation.

## Fixes Applied
1. Added deterministic execution trace artifact:
   - `runs/<run_id>/stage12_trace.json`
   - Counters:
     - `stage12_3.applied_soft_weight_count`
     - `stage12_3.adaptive_usability_samples`
     - `stage12_4.score_computed_count`
     - `stage12_4.threshold_eval_count`
2. Added reject pipeline artifact:
   - `runs/<run_id>/stage12_reject_pipeline.csv`
   - Columns: `combo_id, reason, stage, raw_trade_count, posthook_trade_count, wf_required_trades, wf_actual_trades`
3. Added resolved effective config artifact:
   - `runs/<run_id>/resolved_config.json`
4. Wiring/ordering fixes:
   - Stage-12.4 scoring wrapper is now applied in walkforward holdout/forward paths.
   - Stage-12.3 fallback windows are generated deterministically for short histories.
   - Adaptive min-trades uses **actual forward window duration** (not always 30 days).
   - Effective min usable windows adapt to available windows when Stage-12.3 is enabled.
5. Diagnostic metric semantics fix:
   - `invalid_pct` excludes `ZERO_TRADE` and counts true invalid rejections only.

## Post-Fix Runs
Reference runs:
- Stage-12.3 post-fix: `20260301_060415_b97b463bf5c4_stage12`
- Stage-12.4 post-fix: `20260301_060514_8bb2fcf2c8c0_stage12`

| mode | zero_trade_pct | walkforward_executed_true_pct | MC_trigger_rate | invalid_pct |
| --- | ---: | ---: | ---: | ---: |
| post-fix Stage-12.3 | 62.878788 | 66.666667 | 1.094276 | 31.818182 |
| post-fix Stage-12.4 | 71.969697 | 66.666667 | 0.000000 | 28.030303 |

### Trace counters (post-fix)
- Stage-12.3 run (`20260301_060415_b97b463bf5c4_stage12`):
  - `applied_soft_weight_count=1566`
  - `score_computed_count=0`
  - `threshold_eval_count=0`
- Stage-12.4 run (`20260301_060514_8bb2fcf2c8c0_stage12`):
  - `applied_soft_weight_count=369`
  - `score_computed_count=308`
  - `threshold_eval_count=58212`

This proves Stage-12.3/12.4 codepaths are executed and no longer silent no-op.

### Reject pipeline breakdown (post-fix)
- Stage-12.3 (`20260301_060415_b97b463bf5c4_stage12`)
  - `ZERO_TRADE@prehook=747`
  - `LOW_USABLE_WINDOWS@posthook=378`
  - `VALID@posthook=63`
- Stage-12.4 (`20260301_060514_8bb2fcf2c8c0_stage12`)
  - `ZERO_TRADE@prehook=747`
  - `ZERO_TRADE@posthook=108`
  - `LOW_USABLE_WINDOWS@posthook=333`

Interpretation:
- Prehook `ZERO_TRADE` remains structural for many combinations on current local dataset.
- Posthook path is now active and measurable.

## Added Regression Tests
- `tests/test_stage12_wiring_forensics.py`
  - `test_stage12_trace_has_nonzero_counts_when_enabled`
  - `test_stage12_trace_counts_zero_when_disabled`
  - `test_stage12_walkforward_executes_on_known_trading_combo`
  - `test_stage12_4_scoring_changes_kept_ratio_on_fixture`

## Conclusion
- **FIXED_WIRING**
- Stage-12.3/12.4 hooks are now provably wired and executed.
- Current data regime still shows **no robust edge** under target thresholds (`zero_trade_pct` and `MC_trigger_rate` remain below targets).
