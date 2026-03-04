# Stage-31 Report

## DSL
- Stage-31.1 added an interpretable strategy DSL in `src/buffmini/stage31/dsl.py`.
- Supported operators:
  - Comparators: `>`, `<`, `cross(up/down)`
  - Numeric transforms: `rolling_mean`, `rolling_std`, `rank`, `percentile`
  - Logic: `and`, `or`
- Deterministic and leakage-safe behavior is enforced by tests.

## Evolution
- Pending Stage-31.2.

## Scheduling
- Pending Stage-31.3.

## Runner
- Pending Stage-31.4.

