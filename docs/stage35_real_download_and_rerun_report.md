# Stage-35 Real Download and Rerun Report

Generated: 2026-03-06T00:28:29.149754+00:00

## 1) Key Resolution + Verify
- key_present: `True`
- key_source: `SECRETS_TXT`
- verify_ok: `True`
- verify_http_status: `200`

## 2) Plan Stats
- selected_count: `836` / max_requests `1500`
- truncated: `False`
- symbols: `['BTC/USDT', 'ETH/USDT']`
- endpoints: `['funding_rates', 'open_interest']`
- years: `4`

## 3) Real Download Outcome
- run_id: `20260306_000224_5170837df92b_stage35`
- success_count: `836`
- failure_count: `0`
- usage_summary_path: `runs/20260306_000224_5170837df92b_stage35/coinapi/usage_summary.json`

## 4) CoinAPI Usage Ledger
- total_requests: `836`
- successful_requests: `836`
- failed_requests: `0`
- status_code_counts: `{'200': 836}`
- requests_by_endpoint: `{'funding_rates': 418, 'open_interest': 418}`
- requests_by_symbol: `{'BTC/USDT': 418, 'ETH/USDT': 418}`
- retry_counts: `{'total': 0, 'by_endpoint': {'funding_rates': 0, 'open_interest': 0}}`
- rate_limit_hits: `0`
- bytes_downloaded: `1672`
- estimated_credit_usage: `UNKNOWN`

## 5) Coverage Achieved (Years)
- BTC/USDT funding_rates: `0.010951` years (pass=False)
- BTC/USDT open_interest: `0.000000` years (pass=False)
- ETH/USDT funding_rates: `0.000000` years (pass=False)
- ETH/USDT open_interest: `0.000000` years (pass=False)

## 6) Data Quality + Alignment/Leakage Checks
- BTC/USDT funding: samples=3, dup=0, monotonic=True, gaps=2, max_gap_hours=48.0
- BTC/USDT open_interest: samples=0, dup=0, monotonic=True, gaps=0, max_gap_hours=0.0
- BTC/USDT aligned_non_null funding=2163, oi=0, missing_ratio funding=0.998972, oi=1.000000
- BTC/USDT leakage future-match checks: {'funding_rates_future_matches': 0, 'open_interest_future_matches': 0}
- ETH/USDT funding: samples=0, dup=0, monotonic=True, gaps=0, max_gap_hours=0.0
- ETH/USDT open_interest: samples=0, dup=0, monotonic=True, gaps=0, max_gap_hours=0.0
- ETH/USDT aligned_non_null funding=0, oi=0, missing_ratio funding=1.000000, oi=1.000000
- ETH/USDT leakage future-match checks: {'funding_rates_future_matches': 0, 'open_interest_future_matches': 0}

## 7) Stage-35 Orchestrator + Engine Rerun
- stage35_status: `INSUFFICIENT_COVERAGE`
- stage35_coinapi_enabled: `False`
- stage35_download_attempted: `False`
- research/live rerun executed: `False`
- reason_not_executed: `coverage_threshold_not_met`

## 8) Root Cause + Evidence
- Initial 400 root cause fixed: `period_id is required` now supplied per endpoint.
- After fix, requests succeed (`200`) but return empty arrays across planned slices (request cost header observed as `0` in probes).
- provider_probe: `{'period_required_error_seen': True, 'period_fix_applied': True, 'post_fix_probe': {'http_status': 200, 'payload_type': 'list', 'payload_len': 0, 'request_cost_header': '0'}, 'interpretation': 'Endpoint responds but returns empty datasets for requested futures funding/open-interest slices (likely provider endpoint availability/entitlement/mapping limitation).'}`
- mapping_audit: `docs/stage35_endpoint_mapping_audit.md` (`16` tested mapping combinations, all payload_len=`0`)

## 9) Verdict
- verdict: `INSUFFICIENT_COVERAGE`
- biggest_blocker: `CoinAPI futures funding/open-interest history calls return HTTP 200 with empty datasets for all planned BTC/ETH slices.`

## 10) Next Actions
1. Run a provider capability check for these futures endpoints/symbol mappings with CoinAPI support and account plan details.
2. Switch Stage-35 extras source for funding/open-interest to an alternate provider with guaranteed historical coverage for BINANCE BTC/ETH perpetuals.
3. After source fix, rerun: python scripts/update_coinapi_extras.py --endpoints funding,oi --symbols BTC/USDT,ETH/USDT --years 4 --max-requests 1500
