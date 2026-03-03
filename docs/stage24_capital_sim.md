# Stage-24 Capital Simulation

- run_id: `20260303_023152_a32df3c5e388_stage24_capital`
- seed: `42`
- mode: `risk_pct`
- dry_run: `False`
- base_timeframe: `1m`
- operational_timeframe: `1h`
- symbols: `['BTC/USDT', 'ETH/USDT']`

## Results
| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_order_pct | top_invalid_reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100.00 | 100.000000 | 0.000000 | 0.000000 | 0.00 | 0.000000 | 0.000000 | 0.000000 | VALID |
| 1000.00 | 1000.000000 | 0.000000 | 0.000000 | 0.00 | 0.000000 | 0.000000 | 0.000000 | VALID |
| 10000.00 | 10000.000000 | 0.000000 | 0.000000 | 0.00 | 0.000000 | 0.000000 | 0.000000 | VALID |
| 100000.00 | 100000.000000 | 0.000000 | 0.000000 | 0.00 | 0.000000 | 0.000000 | 0.000000 | VALID |

## Scale Invariance Check
- return_pct_std: `0.000000`
- scale_invariance_ok: `True`
- note: `returns are broadly scale-consistent`

## Artifacts
- results_csv: `runs/20260303_023152_a32df3c5e388_stage24_capital/stage24/capital_sim_results.csv`
- results_json: `runs/20260303_023152_a32df3c5e388_stage24_capital/stage24/capital_sim_results.json`
