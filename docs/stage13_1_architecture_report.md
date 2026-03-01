# Stage-13.1 Report

## 1) What changed
- Stage-13 family engine contract and evaluation layer executed.
- Family runs are deterministic and artifact-driven.

## 2) How to run (dry-run + real)
- dry-run: `python scripts/run_stage13.py --dry-run --seed 42`
- real: `python scripts/run_stage13.py --seed 42`

## 3) Validation gates & results
- zero_trade_pct: `50.000000`
- invalid_pct: `100.000000`
- walkforward_executed_true_pct: `0.000000`
- mc_trigger_rate: `50.000000`

## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)
| symbol | family | composer | trade_count | tpm | PF | expectancy | exp_lcb | maxDD | wf_class | mc | class |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| BTC/USDT | volatility | none | 0.00 | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | INSUFFICIENT_DATA | False | INSUFFICIENT_DATA |
| BTC/USDT | combined | weighted_sum | 0.00 | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | INSUFFICIENT_DATA | False | INSUFFICIENT_DATA |
| ETH/USDT | volatility | none | 0.00 | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | INSUFFICIENT_DATA | False | INSUFFICIENT_DATA |
| ETH/USDT | combined | weighted_sum | 0.00 | 0.00 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | INSUFFICIENT_DATA | False | INSUFFICIENT_DATA |
| ETH/USDT | price | none | 10.00 | 3.00 | 1.2899 | 23.292706 | -39.077691 | 0.066743 | INSUFFICIENT_DATA | True | INSUFFICIENT_DATA |
| ETH/USDT | flow | none | 43.00 | 12.91 | 0.9053 | -10.141276 | -42.616019 | 0.141918 | INSUFFICIENT_DATA | True | INSUFFICIENT_DATA |
| BTC/USDT | flow | none | 52.00 | 15.61 | 0.5616 | -47.836707 | -71.589844 | 0.344839 | INSUFFICIENT_DATA | True | INSUFFICIENT_DATA |
| BTC/USDT | price | none | 20.00 | 6.00 | 0.6928 | -34.820259 | -80.704093 | 0.155220 | INSUFFICIENT_DATA | True | INSUFFICIENT_DATA |

## 5) Failures + reasons
- all_configs_invalid
- walkforward_not_executed

## 6) Next actions
- Continue to next Stage-13/14 sub-stage refinement.

## Repro
- run_id: `20260301_140450_0941cd201d3c_stage13`
- seed: `42`
- config_hash: `18424aca2b5d`
- data_hash: `333ea89bbcdcdac8`
- classification: `NO_EDGE`
