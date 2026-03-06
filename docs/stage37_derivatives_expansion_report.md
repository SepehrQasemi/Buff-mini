# Stage-37 Derivatives Expansion Report

## Availability
- funding available: `True`
- taker buy/sell available: `True`
- long/short ratio available: `True`
- OI short-only mode enabled: `False`

## Coverage by Symbol/Family
| symbol | family | source | sample_count | start_ts | end_ts | coverage_years |
| --- | --- | --- | ---: | --- | --- | ---: |
| BTC/USDT | funding | binance_public_futures | 35061 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| BTC/USDT | open_interest | binance_public_futures | 35062 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| BTC/USDT | taker_buy_sell | binance_public_futures | 35061 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| BTC/USDT | long_short_ratio | binance_public_futures | 35061 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| ETH/USDT | funding | binance_public_futures | 35061 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| ETH/USDT | open_interest | binance_public_futures | 35062 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| ETH/USDT | taker_buy_sell | binance_public_futures | 35061 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |
| ETH/USDT | long_short_ratio | binance_public_futures | 35061 | 2022-03-03T12:00:00+00:00 | 2026-03-03T10:00:00+00:00 | 3.9998 |

## Notes
- Funding is treated as long-history core context.
- OI remains short-horizon only and is not mandatory for long-history training.
- Taker and long/short families use free public Binance futures endpoints.
