# Stage-24 Capital Simulation

- run_id: `20260303_124247_a32df3c5e388_stage24_capital`
- seed: `42`
- mode: `risk_pct`
- dry_run: `True`
- base_timeframe: `1m`
- operational_timeframe: `1h`
- symbols: `['BTC/USDT', 'ETH/USDT']`

## Results
| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_order_pct | top_invalid_reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100.00 | 72.547722 | -27.452278 | 0.204381 | 32.00 | 99.220133 | 0.190000 | 0.000000 | VALID |
| 1000.00 | 725.477221 | -27.452278 | 0.204381 | 32.00 | 992.197745 | 0.076276 | 0.000000 | VALID |
| 10000.00 | 7254.772213 | -27.452278 | 0.204381 | 32.00 | 9921.969947 | 0.024741 | 0.000000 | VALID |
| 100000.00 | 72547.722127 | -27.452278 | 0.204381 | 32.00 | 99219.700545 | 0.020000 | 0.000000 | VALID |

## Scale Invariance Check
- return_pct_std: `0.000000`
- scale_invariance_ok: `True`
- note: `returns are broadly scale-consistent`

## Artifacts
- results_csv: `runs/20260303_124247_a32df3c5e388_stage24_capital/stage24/capital_sim_results.csv`
- results_json: `runs/20260303_124247_a32df3c5e388_stage24_capital/stage24/capital_sim_results.json`
