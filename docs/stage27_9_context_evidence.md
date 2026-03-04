# Stage-27.9 Context Evidence

## Classification Rules
- ROBUST_CONTEXT_EDGE: occurrences >= 50, trades >= 30, exp_lcb > 0, positive_windows_ratio >= 0.55
- WEAK_CONTEXT_EDGE: partial positive evidence with non-trivial support
- NOISE: evidence below thresholds

## Counts
- ROBUST_CONTEXT_EDGE: `0`
- WEAK_CONTEXT_EDGE: `0`
- NOISE: `17`

## Top Context Edges
| symbol | timeframe | context | rulelet | windows | positive_ratio | exp_lcb_median | trades_median | class |
|---|---|---|---|---:|---:|---:|---:|---|
| BTC/USDT | 1h | CHOP | ChopFilterGate | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | CHOP | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | TREND | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | TREND | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | VOL_EXPANSION | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | TREND | MomentumBurst | 1 | 0.000 | -53.278978 | 63 | NOISE |
| BTC/USDT | 1h | VOL_EXPANSION | VolExpansionContinuation | 1 | 0.000 | -53.352542 | 89 | NOISE |
| BTC/USDT | 1h | TREND | StructureBreak | 1 | 0.000 | -54.538190 | 36 | NOISE |
| BTC/USDT | 1h | TREND | TrendPullback | 1 | 0.000 | -61.156942 | 68 | NOISE |
| BTC/USDT | 1h | VOL_COMPRESSION | VolCompressionBreakout | 1 | 0.000 | -72.215018 | 20 | NOISE |
| BTC/USDT | 1h | RANGE | RangeFade | 1 | 0.000 | -75.423614 | 64 | NOISE |
| BTC/USDT | 1h | RANGE | BollingerSnapBack | 1 | 0.000 | -78.259521 | 60 | NOISE |
| BTC/USDT | 1h | RANGE | MeanRevertAfterSpike | 1 | 0.000 | -89.045659 | 59 | NOISE |
| BTC/USDT | 1h | VOL_EXPANSION | FailedBreakReversal | 1 | 0.000 | -101.052337 | 39 | NOISE |
| BTC/USDT | 1h | RANGE | FailedBreakReversal | 1 | 0.000 | -127.741963 | 30 | NOISE |
