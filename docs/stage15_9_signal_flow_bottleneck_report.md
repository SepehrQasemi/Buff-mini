# Stage-15.9 Signal Flow Bottleneck Report

## 1) Executive Summary
- pre_fix run_id: `20260303_023209_818b74392bb2_stage15_9_trace`
- post_fix run_id: `20260303_023209_818b74392bb2_stage15_9_trace`
- top bottlenecks: `[{'gate': 'death_execution', 'death_rate': 0.8457127291950804}, {'gate': 'death_orders', 'death_rate': 0.7390677349861023}, {'gate': 'death_context', 'death_rate': 0.6693877551020408}, {'gate': 'death_confirm', 'death_rate': 0.6693877551020408}, {'gate': 'death_riskgate', 'death_rate': 0.6693877551020408}]`

## 2) System Flow Diagram
- raw -> context -> confirm -> riskgate -> orders -> trades -> WF -> MC

## 3) Bottleneck Tables
### Overall
- death_execution: death_rate=0.845713
- death_orders: death_rate=0.739068
- death_context: death_rate=0.669388
- death_confirm: death_rate=0.669388
- death_riskgate: death_rate=0.669388

### Per Stage
- stage=18 top_gate=death_orders death_rate=1.000000
- stage=20 top_gate=death_orders death_rate=1.000000
- stage=21 top_gate=death_orders death_rate=1.000000
- stage=15 top_gate=death_execution death_rate=0.835109
- stage=16 top_gate=death_execution death_rate=0.835109
- stage=22 top_gate=death_execution death_rate=0.832418
- stage=17 top_gate=death_execution death_rate=0.801254
- stage=classic top_gate=death_context death_rate=0.800000
- stage=19 top_gate=death_context death_rate=0.666667

### Per Timeframe
- tf=4h top_gate=death_execution death_rate=0.931973
- tf=1h top_gate=death_execution death_rate=0.855280
- tf=2h top_gate=death_execution death_rate=0.819242
- tf=15m top_gate=death_execution death_rate=0.816450
- tf=30m top_gate=death_execution death_rate=0.805619

### Per Family
- family=price top_gate=death_context death_rate=1.000000
- family=volatility top_gate=death_context death_rate=1.000000
- family=combined top_gate=death_execution death_rate=0.846665
- family=classic_trend_pullback top_gate=death_context death_rate=0.800000
- family=flow top_gate=death_execution death_rate=0.690034

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
