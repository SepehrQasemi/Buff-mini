# Stage-24 Capital Simulation

- run_id: `20260302_163313_a32df3c5e388_stage24_capital`
- seed: `42`
- mode: `risk_pct`
- dry_run: `True`
- base_timeframe: `1m`
- operational_timeframe: `1h`
- symbols: `['BTC/USDT', 'ETH/USDT']`

## Results
| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_order_pct | top_invalid_reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100.00 | 72.547722 | -27.452278 | 0.204381 | 32.00 | 98.885994 | 0.190000 | 0.000000 | VALID |
| 1000.00 | 725.477221 | -27.452278 | 0.204381 | 32.00 | 994.539140 | 0.076192 | 0.000000 | VALID |
| 10000.00 | 10000.000000 | 0.000000 | 0.000000 | 0.00 | 10000.000000 | 0.025298 | 100.000000 | MARGIN_FAIL |
| 100000.00 | 100000.000000 | 0.000000 | 0.000000 | 0.00 | 100000.000000 | 0.020000 | 100.000000 | MARGIN_FAIL |

## Scale Invariance Check
- return_pct_std: `13.726139`
- scale_invariance_ok: `False`
- note: `high dispersion suggests min_notional/cap effects`

## Artifacts
- results_csv: `runs/20260302_163313_a32df3c5e388_stage24_capital/stage24/capital_sim_results.csv`
- results_json: `runs/20260302_163313_a32df3c5e388_stage24_capital/stage24/capital_sim_results.json`
