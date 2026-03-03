# Stage-24 Capital Simulation

- run_id: `20260303_045724_a32df3c5e388_stage24_capital`
- seed: `42`
- mode: `risk_pct`
- dry_run: `False`
- base_timeframe: `1m`
- operational_timeframe: `1h`
- symbols: `['BTC/USDT', 'ETH/USDT']`

## Results
| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_order_pct | top_invalid_reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100.00 | 81.421908 | -18.578092 | 0.315726 | 239.00 | 91.802903 | 0.169233 | 22.061856 | POLICY_CAP_HIT |
| 1000.00 | 710.789079 | -28.921092 | 0.413387 | 318.00 | 920.587341 | 0.074478 | 0.000000 | VALID |
| 10000.00 | 7107.890792 | -28.921092 | 0.413387 | 318.00 | 9184.710228 | 0.023773 | 0.000000 | VALID |
| 100000.00 | 71078.907922 | -28.921092 | 0.413387 | 318.00 | 91711.887663 | 0.020000 | 0.000000 | VALID |

## Scale Invariance Check
- return_pct_std: `4.478650`
- scale_invariance_ok: `True`
- note: `returns are broadly scale-consistent`

## Artifacts
- results_csv: `runs/20260303_045724_a32df3c5e388_stage24_capital/stage24/capital_sim_results.csv`
- results_json: `runs/20260303_045724_a32df3c5e388_stage24_capital/stage24/capital_sim_results.json`
