# Stage-8 Walk-Forward V2 Audit

- synthetic classification: `INSUFFICIENT_DATA` (2 usable / 4 total)
- synthetic determinism: `True` (hash=d7d0ba5debac)
- synthetic no-future-safe: `True`
- real classification: `UNSTABLE` (12 usable / 16 total)
- real determinism: `True` (hash=7431548bd6fd)
- tightened-threshold classification change observed: `True`
- window overlap checks: holdout/forward=`True` forward-sequence=`True`
- reserve_tail respected: `True`
- finite metrics propagated safely: `True`
- anomalies: `[]`

## Excluded Reasons Histogram

- synthetic: `{'min_trades': 2}`
- real: `{'min_trades': 4}`
