# Stage-15.9 Signal Flow Bottleneck Report

## 1) Executive Summary
- pre_fix run_id: `20260302_214403_ad295ccc2c90_stage15_9_trace`
- post_fix run_id: `20260302_214403_ad295ccc2c90_stage15_9_trace`
- top bottlenecks: `[{'gate': 'death_execution', 'death_rate': 0.7861116941302039}, {'gate': 'death_orders', 'death_rate': 0.6269889110082814}, {'gate': 'death_context', 'death_rate': 0.4897959183673469}, {'gate': 'death_confirm', 'death_rate': 0.4897959183673469}, {'gate': 'death_riskgate', 'death_rate': 0.4897959183673469}]`

## 2) System Flow Diagram
- raw -> context -> confirm -> riskgate -> orders -> trades -> WF -> MC

## 3) Bottleneck Tables
### Overall
- death_execution: death_rate=0.786112
- death_orders: death_rate=0.626989
- death_context: death_rate=0.489796
- death_confirm: death_rate=0.489796
- death_riskgate: death_rate=0.489796

### Per Stage
- stage=18 top_gate=death_orders death_rate=1.000000
- stage=20 top_gate=death_orders death_rate=1.000000
- stage=21 top_gate=death_orders death_rate=1.000000
- stage=15 top_gate=death_execution death_rate=0.701384
- stage=16 top_gate=death_execution death_rate=0.701384
- stage=22 top_gate=death_execution death_rate=0.700666
- stage=17 top_gate=death_execution death_rate=0.647336
- stage=19 top_gate=death_execution death_rate=0.628613
- stage=classic top_gate=death_execution death_rate=0.243182

### Per Timeframe
- tf=15m top_gate=death_execution death_rate=0.786112
- tf=1h top_gate=death_execution death_rate=0.786112
- tf=2h top_gate=death_execution death_rate=0.786112
- tf=30m top_gate=death_execution death_rate=0.786112
- tf=4h top_gate=death_execution death_rate=0.786112

### Per Family
- family=volatility top_gate=death_context death_rate=1.000000
- family=combined top_gate=death_execution death_rate=0.849529
- family=price top_gate=death_execution death_rate=0.708160
- family=flow top_gate=death_execution death_rate=0.576126
- family=classic_trend_pullback top_gate=death_execution death_rate=0.243182

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
