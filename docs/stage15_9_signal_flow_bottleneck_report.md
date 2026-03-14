# Stage-15.9 Signal Flow Bottleneck Report

## 1) Executive Summary
- pre_fix run_id: `20260303_070040_3147eb834b86_stage15_9_trace`
- post_fix run_id: `20260303_070040_3147eb834b86_stage15_9_trace`
- top bottlenecks: `[{'gate': 'death_execution', 'death_rate': 0.6722326038160066}, {'gate': 'death_orders', 'death_rate': 0.49173597221637405}, {'gate': 'death_context', 'death_rate': 0.46904761904761905}, {'gate': 'death_confirm', 'death_rate': 0.4583333333333333}, {'gate': 'death_riskgate', 'death_rate': 0.4583333333333333}]`

## 2) System Flow Diagram
- raw -> context -> confirm -> riskgate -> orders -> trades -> WF -> MC

## 3) Bottleneck Tables
### Overall
- death_execution: death_rate=0.672233
- death_orders: death_rate=0.491736
- death_context: death_rate=0.469048
- death_confirm: death_rate=0.458333
- death_riskgate: death_rate=0.458333

### Per Stage
- stage=18 top_gate=death_orders death_rate=1.000000
- stage=15 top_gate=death_execution death_rate=0.678856
- stage=16 top_gate=death_execution death_rate=0.675770
- stage=17 top_gate=death_execution death_rate=0.634535
- stage=classic top_gate=death_execution death_rate=0.509684

### Per Timeframe
- tf=30m top_gate=death_execution death_rate=0.679784
- tf=1h top_gate=death_execution death_rate=0.676215
- tf=15m top_gate=death_execution death_rate=0.662272

### Per Family
- family=volatility top_gate=death_context death_rate=1.000000
- family=combined top_gate=death_execution death_rate=0.663497
- family=price top_gate=death_execution death_rate=0.643996
- family=classic_trend_pullback top_gate=death_execution death_rate=0.509684
- family=flow top_gate=death_execution death_rate=0.450238

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
