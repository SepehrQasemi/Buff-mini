# CI Repair Report

## Summary
- Date: 2026-02-27
- Scope: Reproduce and fix CI parity failures in a clean, non-editable install environment.

## Reproduction (Clean Environment)
- Created fresh venv: `.venv_ci_test`
- Installed project non-editable: `pip install .`
- Ran:
  - `python -m compileall src scripts tests`
  - `pytest -q`

## Failures Observed
1. `tests/test_trade_map_data_contract.py` (import error at collection)
   - Error: `ModuleNotFoundError: No module named 'matplotlib'`
2. `tests/test_stage0_dry_run.py::test_stage0_dry_run_artifacts_have_required_files_and_shape`
3. `tests/test_stage0_dry_run.py::test_stage0_dry_run_is_reproducible_for_same_config_seed_and_data`
   - Error: `FileNotFoundError` for `.../.venv_ci_test/Lib/configs/default.yaml`

## Root Causes
1. `matplotlib` was used by UI trade-map code/tests but missing from `pyproject.toml` dependencies.
2. Path detection assumed editable/source layout via `Path(__file__).resolve().parents[2]`.
   - In non-editable installs, this points inside the virtualenv instead of repository root.
   - This breaks default config resolution for subprocess-based script tests.

## Fixes Applied
1. Added `matplotlib` to `[project].dependencies` in `pyproject.toml`.
2. Updated `src/buffmini/constants.py`:
   - Added robust `_detect_project_root()`:
     - supports optional `BUFFMINI_PROJECT_ROOT`
     - searches current working directory (and ancestors) for `pyproject.toml` + `configs/default.yaml`
     - falls back to source-relative layout
3. Updated `.github/workflows/ci.yml` for clean-install parity and explicit checks:
   - `pip install .` (non-editable)
   - `python -m compileall src scripts tests`
   - `pytest -q`

## Verification After Fix
- Clean venv (non-editable install):
  - `python -m compileall src scripts tests` -> pass
  - `pytest -q` -> `120 passed`
- Repo environment:
  - `python -m compileall src scripts tests` -> pass
  - `pytest -q` -> `120 passed`

## Outcome
- CI root causes fixed without weakening tests.
- Workflow now matches clean-environment behavior and catches packaging/path regressions.
