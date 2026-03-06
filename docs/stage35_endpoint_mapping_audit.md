# Stage-35 Endpoint Mapping Audit

- window_start: `2026-02-04T00:29:35+00:00`
- window_end: `2026-03-06T00:29:35+00:00`

| combo | endpoint | status | payload_len | request_cost_header |
| --- | --- | ---: | ---: | --- |
| symbol_perp | funding_rates | 200 | 0 | 0 |
| symbol_perp | open_interest | 200 | 0 | 0 |
| symbol_fts | funding_rates | 200 | 0 | 0 |
| symbol_fts | open_interest | 200 | 0 | 0 |
| asset_pair_usdt | funding_rates | 200 | 0 | 0 |
| asset_pair_usdt | open_interest | 200 | 0 | 0 |
| asset_pair_usd | funding_rates | 200 | 0 | 0 |
| asset_pair_usd | open_interest | 200 | 0 | 0 |
| exchange_pair_binance | funding_rates | 200 | 0 | 0 |
| exchange_pair_binance | open_interest | 200 | 0 | 0 |
| exchange_pair_binancefts | funding_rates | 200 | 0 | 0 |
| exchange_pair_binancefts | open_interest | 200 | 0 | 0 |
| exchange_symbol_binance | funding_rates | 200 | 0 | 0 |
| exchange_symbol_binance | open_interest | 200 | 0 | 0 |
| exchange_symbol_binancefts | funding_rates | 200 | 0 | 0 |
| exchange_symbol_binancefts | open_interest | 200 | 0 | 0 |

All tested symbol/asset/exchange mappings for BTC futures funding/open-interest returned HTTP 200 with empty payloads.
