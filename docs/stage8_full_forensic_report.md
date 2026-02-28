# Stage-8 Full Forensic Report

## Scope
- Walk-forward v2 forensic validation (synthetic + real local dataset)
- Cost model v2 realism and safety validation
- Leakage harness effectiveness and completeness
- Pipeline integrity and compatibility checks

## Walk-forward V2
- deterministic: `True`
- real classification: `UNSTABLE`
- usable windows: `12`
- anomalies: `[]`

## Cost Model V2
- finite-safe: `True`
- backward-compatible: `True`
- drag sensitivity: `HIGH`
- simple_vs_v2_delta: `-863.347809`
- delay_impact: `108.758092`

## Leakage Harness
- all registered features clean: `True`
- synthetic leak detected: `True`
- registry complete: `True`

## Pipeline Integrity
- stage0 unchanged under simple mode: `True`
- stage4/stage4.5 unaffected by stage8 toggles (no coupling): `True`
- hash semantics stable: `True`

## CRITICAL_FINDINGS
- none

## Final Verdict
`SAFE_WITH_LIMITATIONS`

Supporting docs:
- docs/stage8_walkforward_audit.md
- docs/stage8_cost_model_audit.md
- docs/stage8_leakage_audit.md
- docs/stage8_pipeline_integrity.md
