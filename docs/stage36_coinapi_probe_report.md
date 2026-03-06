# Stage-36 CoinAPI Probe Report

- generated_at_utc: `2026-03-06T01:08:40.651554+00:00`
- auth_ok: `True`
- key_source: `SECRETS_TXT`
- budget_used_requests: `19` / `50`
- coinapi_available: `True`

## Binance Exchange IDs (from probe)
- `BINANCE`
- `BINANCEDEX`
- `BINANCEFTS`
- `BINANCEFTSC`
- `BINANCEJE`
- `BINANCEJEX`
- `BINANCEOPT`
- `BINANCEOPTV`
- `BINANCEUS`

## Symbol Mapping Choice
- BTC/USDT -> `BINANCE_PERP_BTC_USDT`
- ETH/USDT -> `BINANCE_PERP_ETH_USDT`

## Endpoint Probe Rows
| family | label | symbol | status | payload_len | non_empty | request_cost |
| --- | --- | --- | ---: | ---: | --- | --- |
| metadata | auth_and_exchanges |  | 200 | 409 | True | 1 |
| metadata | list_periods |  | 200 | 2 | True | 1 |
| funding | mapping_probe_BTC/USDT_BINANCE_PERP_BTC_USDT_1d | BTC/USDT | 200 | 0 | False | 0 |
| funding | mapping_probe_BTC/USDT_BINANCEFTS_PERP_BTC_USDT_1d | BTC/USDT | 200 | 0 | False | 0 |
| funding | mapping_probe_ETH/USDT_BINANCE_PERP_ETH_USDT_1d | ETH/USDT | 200 | 0 | False | 0 |
| funding | mapping_probe_ETH/USDT_BINANCEFTS_PERP_ETH_USDT_1d | ETH/USDT | 200 | 0 | False | 0 |
| funding | funding_BTC/USDT_1d | BTC/USDT | 200 | 0 | False | 0 |
| funding | funding_BTC/USDT_7d | BTC/USDT | 200 | 0 | False | 0 |
| funding | funding_ETH/USDT_1d | ETH/USDT | 200 | 0 | False | 0 |
| funding | funding_ETH/USDT_7d | ETH/USDT | 200 | 0 | False | 0 |
| open_interest | open_interest_BTC/USDT_1d | BTC/USDT | 200 | 0 | False | 0 |
| open_interest | open_interest_BTC/USDT_7d | BTC/USDT | 200 | 0 | False | 0 |
| open_interest | open_interest_ETH/USDT_1d | ETH/USDT | 200 | 0 | False | 0 |
| open_interest | open_interest_ETH/USDT_7d | ETH/USDT | 200 | 0 | False | 0 |
| liquidations | liquidations_BTC/USDT_1d | BTC/USDT | 200 | 0 | False | 0 |
| liquidations | liquidations_BTC/USDT_7d | BTC/USDT | 200 | 0 | False | 0 |
| liquidations | liquidations_ETH/USDT_1d | ETH/USDT | 200 | 0 | False | 0 |
| liquidations | liquidations_ETH/USDT_7d | ETH/USDT | 200 | 0 | False | 0 |
