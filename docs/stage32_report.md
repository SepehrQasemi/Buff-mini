# Stage-32 Report

## Pareto Validation
- Stage-32.1 added multi-objective Pareto selection in `src/buffmini/stage32/pareto.py`.
- Objectives:
  - maximize `exp_lcb`
  - maximize `pf_adj`
  - minimize `maxdd_p95`
  - maximize `repeatability`
  - maximize `feasibility_score`

## Nested WF/MC
- Stage-32.2 added nested WF + MC-lite validation runner:
  - script: `scripts/run_stage32_validate.py`
  - module: `src/buffmini/stage32/validate.py`
- Run artifacts:
  - `runs/<run_id>/stage32/validated.csv`
  - `runs/<run_id>/stage32/validated.json`

## Feasibility
- Stage-32.3 integrated feasibility envelope at candidate/policy level.
- Module: `src/buffmini/stage32/feasibility.py`
- Script: `scripts/run_stage32_feasibility.py`
- Run artifacts:
  - `runs/<run_id>/stage32/feasibility_envelope.csv`
  - `runs/<run_id>/stage32/feasibility_summary.json`
