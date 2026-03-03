# Stage-15.9 Signal Flow Bottleneck Report

## 1) Executive Summary
- pre_fix run_id: `20260303_050350_3504ef8051d5_stage15_9_trace`
- post_fix run_id: `20260303_050350_3504ef8051d5_stage15_9_trace`
- top bottlenecks: `[{'gate': 'death_execution', 'death_rate': 0.78726460152795}, {'gate': 'death_orders', 'death_rate': 0.5666666666666667}, {'gate': 'death_context', 'death_rate': 0.55}, {'gate': 'death_confirm', 'death_rate': 0.55}, {'gate': 'death_riskgate', 'death_rate': 0.55}]`

## 2) System Flow Diagram
- raw -> context -> confirm -> riskgate -> orders -> trades -> WF -> MC

## 3) Bottleneck Tables
### Overall
- death_execution: death_rate=0.787265
- death_orders: death_rate=0.566667
- death_context: death_rate=0.550000
- death_confirm: death_rate=0.550000
- death_riskgate: death_rate=0.550000

### Per Stage
- stage=15 top_gate=death_execution death_rate=0.833584
- stage=16 top_gate=death_execution death_rate=0.733834
- stage=classic top_gate=death_execution death_rate=0.669643

### Per Timeframe
- tf=4h top_gate=death_execution death_rate=0.787265

### Per Family
- family=volatility top_gate=death_context death_rate=1.000000
- family=combined top_gate=death_execution death_rate=0.850632
- family=price top_gate=death_execution death_rate=0.730159
- family=classic_trend_pullback top_gate=death_execution death_rate=0.669643
- family=flow top_gate=death_execution death_rate=0.519948

## 4) Findings
- Bug-like findings are listed under Fixes Applied.
- Design bottlenecks are retained and reported without relaxing gates.

## 5) Fixes Applied
- none

## 6) Before/After Deltas
- zero_trade_pct delta: `0.000000`
- invalid_pct delta: `0.000000`
- walkforward_executed_true_pct delta: `0.000000`
- mc_trigger_rate delta: `0.000000`

## 7) Next Steps
- Tune signal-generation density (score/threshold shaping) by family where raw counts collapse.
- Inspect context and risk-gate death rates on weak timeframes before search-space expansion.
- Prioritize timeframes/families with non-zero WF execution before MC-heavy sweeps.
