# Stage-27.9 Context Evidence

## Classification Rules
- ROBUST_CONTEXT_EDGE: occurrences >= 50, trades >= 30, exp_lcb > 0, positive_windows_ratio >= 0.55
- WEAK_CONTEXT_EDGE: partial positive evidence with non-trivial support
- NOISE: evidence below thresholds

## Counts
- ROBUST_CONTEXT_EDGE: `0`
- WEAK_CONTEXT_EDGE: `5`
- NOISE: `165`

## Top Context Edges
| symbol | timeframe | context | rulelet | windows | positive_ratio | exp_lcb_median | trades_median | class |
|---|---|---|---|---:|---:|---:|---:|---|
| BTC/USDT | 15m | CHOP | ChopFilterGate | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 15m | CHOP | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 15m | TREND | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 15m | TREND | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 15m | VOL_EXPANSION | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | CHOP | ChopFilterGate | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | CHOP | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | TREND | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | TREND | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 1h | VOL_EXPANSION | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 2h | CHOP | ChopFilterGate | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 2h | CHOP | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 2h | TREND | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 2h | TREND | TrendFlip | 1 | 0.000 | 0.000000 | 0 | NOISE |
| BTC/USDT | 2h | VOL_EXPANSION | BreakoutRetest | 1 | 0.000 | 0.000000 | 0 | NOISE |
