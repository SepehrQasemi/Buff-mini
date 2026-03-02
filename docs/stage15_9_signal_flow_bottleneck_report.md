# Stage-15.9 Signal Flow Bottleneck Report

## 1) Executive Summary
- pre_fix run_id: `20260302_020415_0322d25efdcf_stage15_9_trace`
- post_fix run_id: `20260302_020611_0322d25efdcf_stage15_9_trace`
- top bottlenecks: `[{'gate': 'death_execution', 'death_rate': 0.8517782238369271}, {'gate': 'death_orders', 'death_rate': 0.7298257702073141}, {'gate': 'death_context', 'death_rate': 0.6530612244897959}, {'gate': 'death_confirm', 'death_rate': 0.6530612244897959}, {'gate': 'death_riskgate', 'death_rate': 0.6530612244897959}]`

## 2) System Flow Diagram
- raw -> context -> confirm -> riskgate -> orders -> trades -> WF -> MC

## 3) Bottleneck Tables
### Overall
- death_execution: death_rate=0.851778
- death_orders: death_rate=0.729826
- death_context: death_rate=0.653061
- death_confirm: death_rate=0.653061
- death_riskgate: death_rate=0.653061

### Per Stage
- stage=18 top_gate=death_orders death_rate=1.000000
- stage=20 top_gate=death_orders death_rate=1.000000
- stage=21 top_gate=death_orders death_rate=1.000000
- stage=15 top_gate=death_execution death_rate=0.844716
- stage=16 top_gate=death_execution death_rate=0.844716
- stage=22 top_gate=death_execution death_rate=0.841514
- stage=classic top_gate=death_context death_rate=0.800000
- stage=17 top_gate=death_execution death_rate=0.798136
- stage=19 top_gate=death_context death_rate=0.650000

### Per Timeframe
- tf=4h top_gate=death_execution death_rate=0.937415
- tf=1h top_gate=death_execution death_rate=0.848359
- tf=2h top_gate=death_execution death_rate=0.834667
- tf=30m top_gate=death_execution death_rate=0.825664
- tf=15m top_gate=death_execution death_rate=0.812786

### Per Family
- family=volatility top_gate=death_context death_rate=1.000000
- family=price top_gate=death_execution death_rate=0.910849
- family=combined top_gate=death_execution death_rate=0.855902
- family=classic_trend_pullback top_gate=death_context death_rate=0.800000
- family=flow top_gate=death_execution death_rate=0.711267

## 4) Findings
- Bug-like findings are listed under Fixes Applied.
- Design bottlenecks are retained and reported without relaxing gates.

## 5) Fixes Applied
- fixed_context_gate_counting_and_death_rate_clamp

## 6) Before/After Deltas
- zero_trade_pct delta: `22.244898`
- invalid_pct delta: `3.265306`
- walkforward_executed_true_pct delta: `0.000000`
- mc_trigger_rate delta: `-14.897959`

## 7) Next Steps
- Tune signal-generation density (score/threshold shaping) by family where raw counts collapse.
- Inspect context and risk-gate death rates on weak timeframes before search-space expansion.
- Prioritize timeframes/families with non-zero WF execution before MC-heavy sweeps.
