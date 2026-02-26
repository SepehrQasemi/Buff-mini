# Stage-0 Cost Sensitivity Report

- Generated (UTC): 2026-02-26T09:25:41Z
- Python: 3.11.9
- Run mode: REAL data (`scripts/run_stage0.py --config ...`), no dry-run
- Base config template: `configs/default.yaml` (same symbols/timeframe/date/seed across all runs)
- Fixed values across matrix: `slippage_pct=0.0005`, `funding_pct_per_day=0.0`

## Run IDs
- cost `1.0` -> `runs/20260226_092426_85a7f2023b3a`
- cost `0.2` -> `runs/20260226_092438_5bf3d8b4c424`
- cost `0.1` -> `runs/20260226_092450_30dbdc5f9765`

## Result Matrix

| cost | symbol | strategy | PF | expectancy | max_drawdown | trade_count | final_equity |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.1 | BTC/USDT | Donchian Breakout | 0.000000 | -7.843137 | 1.000000 | 1275.000000 | 0.000000 |
| 0.2 | BTC/USDT | Donchian Breakout | 0.000000 | -7.843137 | 1.000000 | 1275.000000 | 0.000000 |
| 1.0 | BTC/USDT | Donchian Breakout | 0.000000 | null | null | 1275.000000 | null |
| 0.1 | BTC/USDT | RSI Mean Reversion | 0.000000 | -6.317119 | 1.000000 | 1583.000000 | 0.000000 |
| 0.2 | BTC/USDT | RSI Mean Reversion | 0.000000 | -6.317119 | 1.000000 | 1583.000000 | 0.000000 |
| 1.0 | BTC/USDT | RSI Mean Reversion | 0.000000 | null | null | 1583.000000 | null |
| 0.1 | BTC/USDT | Trend Pullback | 0.000000 | -38.167939 | 1.000000 | 262.000000 | 0.000000 |
| 0.2 | BTC/USDT | Trend Pullback | 0.000000 | -38.167939 | 1.000000 | 262.000000 | 0.000000 |
| 1.0 | BTC/USDT | Trend Pullback | 0.000000 | -3442491879902082264771775388285159242581491028130536796801820132376576.000000 | 90193287253434564991742422628170979813221335983967920874264157224960.000000 | 262.000000 | -901932872534345610838787934201543909994364167506795026464793956567744512.000000 |
| 0.1 | ETH/USDT | Donchian Breakout | 0.000000 | -7.861635 | 1.000000 | 1272.000000 | 0.000000 |
| 0.2 | ETH/USDT | Donchian Breakout | 0.000000 | -7.861635 | 1.000000 | 1272.000000 | 0.000000 |
| 1.0 | ETH/USDT | Donchian Breakout | 0.000000 | null | null | 1272.000000 | null |
| 0.1 | ETH/USDT | RSI Mean Reversion | 0.000000 | -6.361323 | 1.000000 | 1572.000000 | 0.000000 |
| 0.2 | ETH/USDT | RSI Mean Reversion | 0.000000 | -6.361323 | 1.000000 | 1572.000000 | 0.000000 |
| 1.0 | ETH/USDT | RSI Mean Reversion | 0.000000 | null | null | 1572.000000 | null |
| 0.1 | ETH/USDT | Trend Pullback | 0.000000 | -37.037037 | 1.000000 | 270.000000 | 0.000000 |
| 0.2 | ETH/USDT | Trend Pullback | 0.000000 | -37.037037 | 1.000000 | 270.000000 | 0.000000 |
| 1.0 | ETH/USDT | Trend Pullback | 0.000000 | -1920440550536470988297452826039002649977868978296456187229631872153850631684096.000000 | 51851894864484706091095638547789175176788349701060441228837925531983596748800.000000 | 270.000000 | -518518948644847138866734788570027998859548520665084585845490889710681993628352512.000000 |

## Conclusion
- No baseline met `PF > 1` and `expectancy > 0` at `0.2` or `0.1`.
- Phase-2 discovery is unlikely to help without changing assumptions.

## Notes
- Cost `1.0` produced non-finite metrics in some rows due extreme fee assumptions; these are represented as `null` in the JSON summary.
