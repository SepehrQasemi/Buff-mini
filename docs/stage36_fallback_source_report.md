# Stage-36 Fallback Source Report

- CoinAPI branch decision: `BRANCH_C` (derivatives endpoints 200-empty on probe)
- Fallback selected: `scripts/update_futures_extras.py` (Binance USD-M public endpoints via ccxt)
- Execution command: `python scripts/update_futures_extras.py --config configs/default.yaml`

## Fallback Results
- BTC/USDT funding: non_null=35057, coverage_years=3.9993, range=2022-03-03T16:00:00+00:00..2026-03-03T10:00:00+00:00
- BTC/USDT open_interest: non_null=825, coverage_years=0.0940, range=2026-01-28T02:00:00+00:00..2026-03-03T10:00:00+00:00
- BTC/USDT open_interest stop_reason=insufficient_chunk_rows warnings=['Coverage ratio below 0.90; Binance OI history retention likely limited.']
- ETH/USDT funding: non_null=35057, coverage_years=3.9993, range=2022-03-03T16:00:00+00:00..2026-03-03T10:00:00+00:00
- ETH/USDT open_interest: non_null=825, coverage_years=0.0940, range=2026-01-28T02:00:00+00:00..2026-03-03T10:00:00+00:00
- ETH/USDT open_interest stop_reason=insufficient_chunk_rows warnings=['Coverage ratio below 0.90; Binance OI history retention likely limited.']

## Notes
- OI horizon is limited by exchange retention/API behavior (`insufficient_chunk_rows`).
- Funding history is near 4 years and is usable for enrichment.
