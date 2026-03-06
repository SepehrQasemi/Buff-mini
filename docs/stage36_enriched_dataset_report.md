# Stage-36 Enriched Dataset Report

Generated: 2026-03-06T01:11:20.994360+00:00

## Data Families Available
- BTC/USDT funding: provider=binance_public_ccxt, non_null_samples=35057, coverage_years=3.9993, duplicates=0, range=2022-03-03T16:00:00+00:00..2026-03-03T10:00:00+00:00
- BTC/USDT open_interest: provider=binance_public_ccxt, non_null_samples=825, coverage_years=0.0940, duplicates=0, range=2026-01-28T02:00:00+00:00..2026-03-03T10:00:00+00:00
- ETH/USDT funding: provider=binance_public_ccxt, non_null_samples=35057, coverage_years=3.9993, duplicates=0, range=2022-03-03T16:00:00+00:00..2026-03-03T10:00:00+00:00
- ETH/USDT open_interest: provider=binance_public_ccxt, non_null_samples=825, coverage_years=0.0940, duplicates=0, range=2026-01-28T02:00:00+00:00..2026-03-03T10:00:00+00:00

## CoinAPI Families Status
- funding_rates: unavailable (HTTP 200 empty payload in 1d/7d probes)
- open_interest: unavailable (HTTP 200 empty payload in 1d/7d probes)
- liquidations: unavailable (HTTP 200 empty payload in 1d/7d probes)

## Provider Mix in Final Dataset
- OHLCV: existing local canonical dataset
- Derivatives extras: Binance public fallback (funding + open_interest)
- CoinAPI: discovery metadata only (no usable derivatives payload)

## Remaining Gaps
- coinapi_funding_rates_binance_perp
- coinapi_open_interest_binance_perp
- coinapi_liquidations_binance_perp
