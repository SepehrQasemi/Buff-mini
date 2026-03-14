# Executive Summary
- Audited source report: `docs/stage4_5_stage5_6_full_report.md`
- Audited run id: `stage5_2_manual_ui`
- Report includes no explicit `CRITICAL FINDINGS` heading. The only two finding strings consistent with `critical_findings=2` are:
  - `Low weekly forward-step stability`
  - `High sensitivity to execution drag`
- Finding verification:
  - `Low weekly forward-step stability`: `VERIFIED`
  - `High sensitivity to execution drag`: `VERIFIED`
- GitHub state: `CLEAN` and `SYNCED` (`HEAD == origin/main`)
- Determinism: `CONFIRMED`

# Stage-4.5 Audit
## Artifact Check
- `runs/stage5_2_manual_ui/reality_check/reality_check_summary.json`: present
- `runs/stage5_2_manual_ui/reality_check/rolling_forward_steps.csv`: present
- `runs/stage5_2_manual_ui/reality_check/perturbation_table.csv`: present
- `runs/stage5_2_manual_ui/reality_check/execution_drag_table.csv`: present

## Finding 1 Validation
- Finding text: `Low weekly forward-step stability`
- Stage: `4.5`
- Evidence:
  - `rolling_forward_steps.csv` weekly step stability ratio (`expectancy >= 0` and `profit_factor >= 1`) = `0.368421...` (< 0.5 threshold used by scoring logic).
  - `reality_check_summary.json.reasons` contains exact string.
- Status: `VERIFIED`

## Finding 2 Validation
- Finding text: `High sensitivity to execution drag`
- Stage: `4.5`
- Evidence:
  - `execution_drag_table.csv` worst `return_pct = -0.498712...` while baseline return from perturbation baseline case is `0.022308...`.
  - Drag robustness score collapses to `0.0` under scoring rule (high sensitivity).
  - `reality_check_summary.json.reasons` contains exact string.
- Status: `VERIFIED`

## Consistency Checks
- `confidence_score = 0.447368...` and `verdict = WARN` are logically aligned (`0.40 <= score < 0.70`).
- No NaN in numeric metrics across rolling/perturbation/drag tables.
- Baseline invariant rows match original baseline metrics (`noise=0`, `delay=0 & slippage=0`).

# Stage-5.6 Audit
## Artifact Check
- `runs/stage5_2_manual_ui/exports/pine/index.json`: present
- `runs/stage5_2_manual_ui/exports/pine/*.pine.txt`: present (7 components + 1 portfolio template)

## Validation Against Claimed Constraints
- Every exported file includes:
  - `//@version=5`
  - `strategy(`
  - `input.`
  - run id reference in header/input
- No `lookahead_on` and no future-bar negative indexing found.
- `index.json.validation.all_files_valid = true`
- Parameter validation status for all components: `pass`

## Pine Limitation Evidence (as documented warning)
- `portfolio_template.pine.txt` explicitly states:
  - `WARNING: portfolio mode is visual approximation; internal multi-component engine semantics differ.`
- This is a real limitation and is present in artifact code, not only in markdown narrative.

# GitHub Integrity
- `git status`: clean working tree
- Branch: `main`
- `git fetch origin` completed
- `git rev-parse HEAD` == `git rev-parse origin/main`
- Stage commits on remote:
  - Stage-4.5 commit `ebd5419a4e9bf6c908c57165186bb0a386bf91f2` is ancestor of `origin/main`
  - Stage-5.6 commit `249e311ffb1d32764c20fd1e854c771db6b452a8` is ancestor of `origin/main`
- `runs/` ignore status:
  - run artifacts under `runs/*` are ignored
  - only `runs/.gitkeep` is tracked
- `library/` status:
  - tracked intentionally (`library/index.json`, cards/spec metadata files)

# Artifact Consistency
- Required docs exist:
  - `docs/stage4_5_stage5_6_full_report.md`
  - `docs/stage4_5_stage5_6_full_summary.json`
- Summary JSON declares:
  - `critical_findings = 2`
  - `recommendation = SAFE FOR PAPER TRADING`
- Markdown report has no explicit `CRITICAL FINDINGS` section; however it lists exactly two Stage-4.5 reason lines matching the summary count and substance.
- Determinism checks:
  - `reality_check_summary.json` SHA256 recomputed: stable
  - Re-running `scripts/export_pine.py --run-id stage5_2_manual_ui` preserves exported file hash map

# Final Verdict
`SAFE FOR PAPER TRADING`

Notes:
- No unverified critical-finding claim found; both extracted findings are evidenced in Stage-4.5 artifacts.
- Documentation structure issue remains: missing explicit `CRITICAL FINDINGS` heading in the prior full report.
