# Stage-4 Execution Layer

Stage-4 adds an execution-ready layer on top of Stage-1/2/3 research outputs.

Scope:

- Execution policy selection (`net`, `hedge`, `isolated`)
- Global risk controls (gross/net caps, sizing, kill-switch)
- Trading spec generation for bot integration
- Offline paper-execution simulation from local run artifacts

Out of scope:

- Live exchange order routing
- Automatic deployment
- Any change to Stage-1 discovery or Stage-2/3 evaluation math

## Execution Modes

- `net`: opposite signals on the same symbol are netted into one target exposure.
- `hedge`: long/short components can coexist; components are kept separate.
- `isolated`: strategy-level components are isolated while global caps still apply.

## Stage-4 Commands

Generate trading spec (writes docs):

```bash
python scripts/run_stage4_spec.py \
  --stage2-run-id 20260227_015806_3cb775eb81a0_stage2 \
  --stage3-3-run-id 20260227_113410_aca9cf2325a2_stage3_3_selector
```

Run offline paper simulation (writes run artifacts under `runs/`):

```bash
python scripts/run_stage4_simulate.py \
  --stage2-run-id 20260227_015806_3cb775eb81a0_stage2 \
  --stage3-3-run-id 20260227_113410_aca9cf2325a2_stage3_3_selector \
  --days 90 \
  --seed 42
```

## Warnings

- Leverage and tail risk remain regime-dependent.
- Block-bootstrap and historical signal structure do not guarantee future behavior.
- Re-evaluate Stage-1 through Stage-3.3 on schedule (`risk.reeval`) before any policy changes.

