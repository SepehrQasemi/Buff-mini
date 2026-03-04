# Stage-27.9 Rolling Discovery Summary

## Config
- run_id: `20260304_120218_732731ed0966_stage27_9_roll`
- seed: `42`
- dry_run: `True`
- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['15m', '30m', '1h', '2h', '4h']`
- windows: `[3, 6]`
- step_months: `1`

## Window Coverage
- 3m: generated=10, evaluated=10
- 6m: generated=0, evaluated=0

## Metrics
- rows: `170`
- positive_exp_lcb_rows: `5`
- runtime_seconds: `25.055094`

## Top Contextual Rows
| symbol | timeframe | window_months | context | rulelet | trade_count | exp | exp_lcb | pf |
|---|---:|---:|---|---|---:|---:|---:|---:|
| ETH/USDT | 15m | 3 | VOLUME_SHOCK | MomentumBurst | 2 | 176.224912 | 148.429208 | 0.873398 |
| ETH/USDT | 1h | 3 | VOLUME_SHOCK | MomentumBurst | 2 | 176.224912 | 148.429208 | 0.873398 |
| ETH/USDT | 2h | 3 | VOLUME_SHOCK | MomentumBurst | 2 | 176.224912 | 148.429208 | 0.873398 |
| ETH/USDT | 30m | 3 | VOLUME_SHOCK | MomentumBurst | 2 | 176.224912 | 148.429208 | 0.873398 |
| ETH/USDT | 4h | 3 | VOLUME_SHOCK | MomentumBurst | 2 | 176.224912 | 148.429208 | 0.873398 |
| BTC/USDT | 15m | 3 | TREND | BreakoutRetest | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 15m | 3 | VOL_EXPANSION | BreakoutRetest | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 15m | 3 | CHOP | ChopFilterGate | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 15m | 3 | CHOP | TrendFlip | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 15m | 3 | TREND | TrendFlip | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | TREND | BreakoutRetest | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | VOL_EXPANSION | BreakoutRetest | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | CHOP | ChopFilterGate | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | CHOP | TrendFlip | 0 | 0.000000 | 0.000000 | 0.000000 |
| BTC/USDT | 1h | 3 | TREND | TrendFlip | 0 | 0.000000 | 0.000000 | 0.000000 |
