# Stage-33 Report

## Policy Builder v3
- Stage-33.1 implemented contextual+MTF policy composition:
  - `src/buffmini/stage33/policy_v3.py`
- Outputs:
  - run-scoped `policy.json`
  - run-scoped `policy_spec.md`

## Signal Emitter
- Stage-33.2 added local signal emitter:
  - script: `scripts/emit_signals.py`
  - module: `src/buffmini/stage33/emitter.py`
- How to use:
  - `python scripts/emit_signals.py --policy-path runs/<run_id>/stage33/policy.json --symbol BTC/USDT --timeframe 1h`
- Output payload contains:
  - context probabilities
  - action (`LONG`/`SHORT`/`FLAT`)
  - confidence + sizing%
  - stop/exit summary
  - feasibility notes + explanation

## Drift + Master
- Pending Stage-33.3.
