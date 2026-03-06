# Stage-36 Provider Capability Matrix

| endpoint_family | auth_ok | payload_non_empty | symbol_mapping_confirmed | estimated_historical_viability | recommended_action | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| funding | True | False | False | none | skip | BTC/USDT->BINANCE_PERP_BTC_USDT checks=[('1d', 200, 0), ('7d', 200, 0)]; ETH/USDT->BINANCE_PERP_ETH_USDT checks=[('1d', 200, 0), ('7d', 200, 0)] |
| open_interest | True | False | False | none | skip | BTC/USDT->BINANCE_PERP_BTC_USDT checks=[('1d', 200, 0), ('7d', 200, 0)]; ETH/USDT->BINANCE_PERP_ETH_USDT checks=[('1d', 200, 0), ('7d', 200, 0)] |
| liquidations | True | False | False | none | skip | BTC/USDT->BINANCE_PERP_BTC_USDT checks=[('1d', 200, 0), ('7d', 200, 0)]; ETH/USDT->BINANCE_PERP_ETH_USDT checks=[('1d', 200, 0), ('7d', 200, 0)] |
| derivatives_metadata | True | True | n/a | potential | use_for_discovery | metadata_rows=2 |

Rules applied: HTTP 200 + empty payload => NOT USABLE.
