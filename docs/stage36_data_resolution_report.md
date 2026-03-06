# Stage-36 Data Resolution Report

Generated: 2026-03-06T03:05:30.447548+00:00

## 1) CoinAPI Usability on Current Free-Credit Account
- auth_ok: `True`
- discovery_requests_used: `19` / `50`
- funding/open_interest/liquidations probes: HTTP 200 but empty payloads for 1d and 7d windows (treated as unusable).
- provider verdict: `BLOCKED_FOR_REQUIRED_DERIVATIVES`

## 2) Branch Decision
- chosen_branch: `BRANCH_C`
- coinapi derivatives skipped to avoid wasting credits.

## 3) Fallback Used
- source: Binance public endpoints via `scripts/update_futures_extras.py`
- families added: funding + open_interest

## 4) Final Enriched Dataset Achieved
- funding BTC years: `3.9993`
- funding ETH years: `3.9993`
- OI BTC years: `0.0940`
- OI ETH years: `0.0940`

## 5) Engine Baseline vs Enriched
- baseline_verdict: `NO_EDGE`
- enriched_verdict: `NO_EDGE`
- wf_executed_pct: baseline=100.0, enriched=100.0, delta=0.0
- mc_trigger_pct: baseline=100.0, enriched=100.0, delta=0.0
- research_trade_count: baseline=0.0, enriched=0.0, delta=0.0
- live_trade_count: baseline=0.0, enriched=0.0, delta=0.0
- research_best_exp_lcb: baseline=0.0, enriched=0.0, delta=0.0
- live_best_exp_lcb: baseline=0.0, enriched=0.0, delta=0.0

## 6) Biggest Remaining Bottleneck
- Even with near-4y funding fallback, policy activation remains zero trades (signal quality/cost gating bottleneck).

## 7) Cheapest Next Step
- Add one more free derivatives context family (e.g., Binance taker buy/sell volume or long/short ratio) with the same no-leak alignment and rerun Stage-28 baseline vs enriched.

## Final Verdict
- `DATA_RESOLVED_BUT_NO_EDGE`
