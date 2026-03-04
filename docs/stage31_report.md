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
- Pending Stage-31.3.

## Runner
- Pending Stage-31.4.
