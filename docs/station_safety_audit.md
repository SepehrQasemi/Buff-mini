# Repo State
- Branch: `main`
- `HEAD`: `7cc5667dd305a971a5cb7c77929a677116091fd1`
- `origin/main`: `7cc5667dd305a971a5cb7c77929a677116091fd1`
- Sync check: `HEAD == origin/main` (PASS)
- Working tree status: **NOT CLEAN** (untracked local artifacts: `.venv_ci_test/`, `build/`, `docs/post_stage5_forensic_audit.md`)

# Architectural Consistency
## Stage Chain Check
Target chain: Stage-1 -> Stage-2 -> Stage-3.3 -> Stage-4 -> Stage-4.5 -> Stage-5.6 -> UI bundle -> Library

## Verified Consistencies
- Stage-3.3 selected method/leverage (`equal`, `5.0x`) matches:
  - `runs/stage5_2_manual_ui/pipeline_summary.json`
  - `runs/stage5_2_manual_ui/ui_bundle/summary_ui.json`
  - `runs/20260227_113410_aca9cf2325a2_stage3_3_selector/selector_summary.json`
  - `docs/trading_spec.md`
  - `library/strategies/equal_1h_87b3cf75d3/strategy_card.json` (origin run `stage5_2_manual_ui`)
- `ui_bundle` summary fields are consistent with pipeline and selector summaries for method/leverage/key metrics.

## Inconsistencies
1. **Missing Stage-1 pointer in pipeline summary**
   - `runs/stage5_2_manual_ui/pipeline_summary.json` has no `stage1_run_id`.
   - Stage-2 summary still points to Stage-1, so chain is recoverable, but top-level pipeline lineage is incomplete.
2. **Stage-4 spec docs are globally overwritten**
   - `scripts/run_stage4_spec.py` default output is `docs/trading_spec.md`.
   - `scripts/run_pipeline.py` calls Stage-4 spec without run-specific output override.
   - Result: later runs can overwrite prior spec/checklist silently.
3. **Stage-4.5 does not bind directly to Stage-4 policy config**
   - `src/buffmini/execution/reality_check.py` reads `ui_bundle` equity/exposure/trades and fixed stress settings.
   - It does not explicitly ingest Stage-4 leverage/execution mode/caps/cost inputs as audit constraints.
   - This violates a strict “exact same config binding” expectation.

# Determinism
Latest completed run used: `stage5_2_manual_ui`

- Stage-4.5 rerun:
  - `reality_check_summary.json` hash before/after rerun: identical
  - SHA256: `3f4dfe2932e02fadf976a66cef7ea83efc68b7b7993681b4143f70a6e8f3b241`
- Stage-5.6 rerun:
  - All exported Pine file hashes before/after rerun: identical (8 files)
- Determinism status: **CONFIRMED**

# Execution Model
- Fee conversion uses percent semantics correctly:
  - `round_trip_pct -> one_way = (pct / 100) / 2` in `src/buffmini/backtest/costs.py`.
- Fees/slippage application:
  - entry fee applied once at entry
  - exit fee applied once at exit
  - directional slippage applied on each fill (`buy` worse, `sell` worse).
- Stop-first priority:
  - backtest engine checks `stop_hit` before TP on the same bar.
- Signal/lookahead handling:
  - strategy signals are shifted by one bar (`shift(1)` in `generate_signals`).
  - Donchian channels are shifted by one bar in features.
- Execution drag delay:
  - Stage-4.5 drag scenario uses `returns.shift(delay_bars)`; delay=1 is a true one-bar lag.

# Pine Mapping
Trend Pullback mapping check (Python vs Pine):
- EMA fast/slow comparison direction: matches.
- Signal EMA condition (`close > signal_ema` long, `<` short): matches.
- RSI threshold structure (`< long_entry`, `> short_entry`): matches.
- Entry signal shift by one bar: present in Pine (`longSignal = (...) [1]`).
- No lookahead misuse found:
  - no `lookahead_on`
  - no future-bar negative indexing.

## Limitation
- `portfolio_template.pine.txt` is explicitly an approximation and cannot replicate full internal multi-component/multi-symbol execution semantics exactly.

# Risk Model Integrity
- Objective is expected log growth (not raw return):
  - `compute_log_growth` / `summarize_utility` in `src/buffmini/portfolio/leverage_selector.py`.
- Hard constraints enforced per leverage candidate:
  - `max_p_ruin`
  - `max_dd_p95`
  - `min_return_p05`
- Selection path:
  - only rows with `pass_all_constraints=True` enter feasible set.
  - if no feasible rows -> `NO_FEASIBLE_LEVERAGE`.
- No silent bypass path found in Stage-3.3 selection flow.

# Coverage Gaps
Critical module coverage status:
- Backtest engine:
  - Has direct tests (`test_backtest_basic`, `test_backtest_exit_priority`, `test_costs`)
  - Gap: limited coverage for full exit-mode matrix (`trailing_atr`, `partial_then_trail`) and short-side edge cases.
- Leverage selector:
  - Has direct tests (`test_stage3_leverage_selector`)
  - Gap: limited full integration assertions on report/artifact cross-consistency.
- Reality check:
  - Has direct tests (`test_stage4_5_reality_check`)
  - Gap: no test enforcing strict Stage-4 policy/config binding.
- Pine export:
  - Has direct tests (`test_stage5_6_pine_export`)
  - Gap: no full equivalence testing of all exit semantics against Python engine outcomes.

# Inconsistencies
Total inconsistencies: **3**
1. Missing `stage1_run_id` in latest pipeline summary lineage.
2. Global Stage-4 spec docs can be silently overwritten by subsequent runs.
3. Stage-4.5 is not explicitly bound to Stage-4 policy/caps/cost/leverage config.

# Limitations
Total limitations: **3**
1. Pine portfolio template is approximation-only (explicitly documented in artifact).
2. Stage-4.5 robustness layer is return-series stress on bundled outputs, not a full re-execution of Stage-4 risk policy.
3. Some workflows depend on “latest available local data” when end-date is not pinned, so reproducibility across time requires frozen data snapshots.

# FINAL CLASSIFICATION
**STABLE BUT HAS KNOWN LIMITATIONS**
