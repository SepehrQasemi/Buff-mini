# Stage-31 Report

## DSL
- Stage-31.1 added an interpretable strategy DSL in `src/buffmini/stage31/dsl.py`.
- Supported operators:
  - Comparators: `>`, `<`, `cross(up/down)`
  - Numeric transforms: `rolling_mean`, `rolling_std`, `rank`, `percentile`
  - Logic: `and`, `or`
- Deterministic and leakage-safe behavior is enforced by tests.

## Evolution
- Stage-31.2 implemented genetic strategy synthesis in `src/buffmini/stage31/evolve.py`.
- Includes:
  - reproducible random population initialization
  - mutation and crossover operators over DSL trees
  - novelty constraint via signal similarity with max similarity threshold in elite selection

## Scheduling
- Stage-31.3 added deterministic Hyperband/Successive Halving scheduling in `src/buffmini/stage31/hyperband.py`.
- Properties:
  - budgeted rungs with shrinking candidate pools
  - enforced exploration quota at each rung (`>=15%`)
  - reproducible candidate selection from fixed seeds

## Runner
- Stage-31.4 added orchestrator: `scripts/run_stage31_synthesis.py`.
- Run artifacts:
  - `runs/<run_id>/stage31/candidates_top.csv`
  - `runs/<run_id>/stage31/candidates_top.json`
  - `runs/<run_id>/stage31/novelty_stats.json`
