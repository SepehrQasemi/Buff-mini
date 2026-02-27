# Executive Summary
- Reality Check status: `WARN` on run `stage5_2_manual_ui` with `confidence_score=0.447368`.
- Pine Export status: `PASS` on run `stage5_2_manual_ui` (`7` components + portfolio template exported).
- Determinism status: `PASS` for both Stage-4.5 and Stage-5.6 (hashes stable across repeated runs with same seed/inputs).
- Overall safety rating: `SAFE FOR PAPER TRADING`.

# Stage-4.5 Validation
## Scope
- Command used: `python scripts/run_stage4_5_reality_check.py --run-id stage5_2_manual_ui --seed 42`
- Artifact root: `runs/stage5_2_manual_ui/reality_check/`

## Determinism Proof
- Re-ran Stage-4.5 twice with identical seed and run-id.
- `reality_check_summary.json` SHA256 remained identical:
  - `3f4dfe2932e02fadf976a66cef7ea83efc68b7b7993681b4143f70a6e8f3b241`
- Result: `PASS`.

## Schema and Invariant Checks
- Required keys present: `confidence_score`, `verdict`, `reasons`, `rolling_forward`, `perturbation`, `execution_drag`.
- Logical checks:
  - `confidence_score` in `[0,1]`: `PASS` (`0.447368`).
  - Verdict threshold alignment (`PASS/WARN/FAIL` bands): `PASS` (`WARN` band).
  - Baseline perturbation match (`noise=0`): `PASS`.
  - Baseline execution-drag match (`delay=0`, `slippage=0`): `PASS`.
  - Numeric NaN/Inf checks for step/perturbation/drag tables: `PASS`.
  - Silent failure observed: `NO`.

## Stress Test Results
- Rolling forward step count (weekly stepping): `57`.
- Confidence: `0.447368`.
- Verdict: `WARN`.
- Reasons:
  - `Low weekly forward-step stability`
  - `High sensitivity to execution drag`

## Identified Fragility Zones
- Forward consistency is not strong enough for a PASS verdict.
- Execution drag sensitivity is a material weakness in current robustness profile.

# Stage-5.6 Validation
## Scope
- Command used: `python scripts/export_pine.py --run-id stage5_2_manual_ui`
- Artifact root: `runs/stage5_2_manual_ui/exports/pine/`

## Export Coverage
- Exported:
  - `index.json`
  - `cand_*.pine.txt` for all selected components (`7`)
  - `portfolio_template.pine.txt`
- Validation checks per file:
  - Contains `//@version=5`: `PASS`
  - Contains `strategy(...)`: `PASS`
  - Contains `input.*`: `PASS`
  - Header includes run_id: `PASS`
  - No `lookahead_on` and no future bar references: `PASS`

## Determinism Proof
- Export run repeated with same inputs.
- `index.json` SHA256 remained identical:
  - `7efc6dccd592ff441aef1d8133f499b53bebb862cb4298ae8ceaa56eb0b446c1`
- Internal deterministic flag: `True`.
- Result: `PASS`.

## Strategy Mapping Table (Python vs Pine)
| Strategy Family | Python Signal Definition Source | Pine Mapping Status |
| --- | --- | --- |
| DonchianBreakout | `src/buffmini/baselines/stage0.py::_base_conditions` (`close > donchian_high`, `close < donchian_low`) | PASS |
| RSIMeanReversion | `rsi_14 < long_entry`, `rsi_14 > short_entry` | PASS |
| TrendPullback | EMA fast/slow trend + close vs signal EMA + RSI thresholds | PASS |
| BollingerMeanReversion | close vs Bollinger bands + RSI thresholds | PASS |
| RangeBreakoutTrendFilter | Donchian breakout gated by EMA trend direction | PASS |

## Parameter Equivalence Table
- Cross-parameter validation source: `ui_bundle/summary_ui.json -> selected_components`.
- Component status:
  - `cand_001106_0220dce2`: `pass`
  - `cand_000637_42edc0b1`: `pass`
  - `cand_001264_848f658f`: `pass`
  - `cand_001139_d55f8520`: `pass`
  - `cand_001012_dbec7b03`: `pass`
  - `cand_001603_418d4b75`: `pass`
  - `cand_001147_0131018a`: `pass`

## Known Pine Limitations
- TradingView strategy execution cannot exactly reproduce all internal engine fill semantics in all edge cases.
- Multi-component, multi-symbol behavior is approximated in `portfolio_template.pine.txt` (visual scaffold), not a full portfolio engine.
- Exit nuances (partial fills, exact stop-first behavior under intrabar ambiguity) remain approximations.

# Risk Disclosure
- Pine cannot fully replicate internal portfolio-level order matching and execution accounting.
- Stage-4.5 weekly forward emulation is a robustness proxy, not a live deployment guarantee.
- Current Stage-4.5 verdict is `WARN`, indicating non-trivial sensitivity under realistic stress (especially execution drag).

# Final Recommendation
- Recommendation: `SAFE FOR PAPER TRADING`.
- Not promoted to small live test from this audit alone due Stage-4.5 `WARN` profile.
