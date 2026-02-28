# Stage-10.7 Report

## Regime Calibration
- regime_distribution: `{'TREND': 45.87988574321148, 'RANGE': 12.130744476985935, 'VOL_EXPANSION': 17.820081715298116, 'VOL_COMPRESSION': 19.165129985175543, 'CHOP': 5.004158079328922}`
- calibration: `{'single_regime_warning': False, 'warnings': [], 'median_trend_strength': 0.008332611598270034, 'atr_percentile_distribution': {'p05': 0.027777777777777776, 'p50': 0.48412698412698413, 'p95': 0.9722222222222222}}`

## Sandbox Ranking (Real Data)
- sandbox_run_id: `20260228_134824_d58e1aa34355_stage10_sandbox`
- enabled_signals: `['ATR_DistanceRevert', 'BollingerSnapBack', 'BreakoutRetest', 'MA_SlopePullback', 'VolCompressionBreakout']`
- disabled_signals: `['RangeFade']`
- ranking_table: `C:\dev\Buff-mini\runs\20260228_134824_d58e1aa34355_stage10_sandbox\sandbox_rankings.csv`

## Exit A/B Isolation
- exit_ab_run_id: `20260228_135329_642b3888fb30_stage10_exit_ab`
- selected_exit: `atr_trailing`
| exit_mode | trade_count | PF | expectancy | maxDD | exp_lcb | drag_sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atr_trailing | 2919.00 | 0.261619 | -3.285538 | 0.985543 | -3.961143 | 0.000000 |
| fixed_atr | 1738.00 | 0.570098 | -5.128981 | 0.978662 | -7.879517 | 0.000000 |

## Before/After Comparison
### Dry
- Stage-9 baseline run-context: `20260228_122912_9a27f1e5886a_stage10`
- Stage-10.6 run_id: `20260228_111937_2d4caf91e389_stage10`
- Stage-10.7 run_id: `20260228_122912_9a27f1e5886a_stage10`
- walkforward: `N/A -> N/A`
- usable_windows: `0 -> 0`

### Real
- available: `True`
- Stage-10.6 run_id: `20260228_134946_d1c6cc662deb_stage10`
- Stage-10.7 run_id: `20260228_135215_8502c83e35d7_stage10`
- walkforward: `UNSTABLE -> UNSTABLE`
- usable_windows: `16 -> 16`

| metric | Stage-9 baseline | Stage-10.6 | Stage-10.7 |
| --- | ---: | ---: | ---: |
| trade_count | 6275.000000 | 1738.000000 | 2919.000000 |
| profit_factor | 0.616272 | 0.570098 | 0.261619 |
| expectancy | -7.830390 | -5.128981 | -3.285538 |
| max_drawdown | 0.954500 | 0.978662 | 0.985543 |
| pf_adj | 0.619305 | 0.582120 | 0.274054 |
| exp_lcb | -7.896892 | -7.879517 | -3.961143 |

## Trade Count Guard
- pass: `False`
- observed_drop_pct: `53.482072`
- max_drop_pct: `10.00`
- family_reduction_breakdown (top 5 by delta_vs_baseline_total):
  - BreakoutRetest: trade_count=0.00, delta_vs_baseline_total=-6275.00
  - VolCompressionBreakout: trade_count=6.00, delta_vs_baseline_total=-6269.00
  - RangeFade: trade_count=248.00, delta_vs_baseline_total=-6027.00
  - MA_SlopePullback: trade_count=958.00, delta_vs_baseline_total=-5317.00
  - ATR_DistanceRevert: trade_count=5627.00, delta_vs_baseline_total=-648.00

## Determinism
- pass: `True`
- detail: `{'pass': True, 'notes': 'PASS', 'signature_previous': '29e4517119c707f17a764e3c', 'signature_latest': '29e4517119c707f17a764e3c'}`

## Final Verdict
- REGRESSION
