# Stage-9 OI Backfill Report

| symbol | old_start | new_start | row_count | expected_rows | coverage_ratio | gap_count | largest_gap_hours | stop_reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BTC/USDT | None | 2026-01-28T03:00:00+00:00 | 703 | 27658 | 0.025418 | 0 | 1.000 | insufficient_chunk_rows |
| ETH/USDT | None | 2026-01-28T03:00:00+00:00 | 703 | 27658 | 0.025418 | 0 | 1.000 | insufficient_chunk_rows |

## Improvement vs Previous Meta
- BTC/USDT: old_coverage=0.000000, new_coverage=0.025418, delta=0.025418
- BTC/USDT: old_rows=0, new_rows=703
- ETH/USDT: old_coverage=0.000000, new_coverage=0.025418, delta=0.025418
- ETH/USDT: old_rows=0, new_rows=703

## Notes
- Coverage may remain low if Binance OI API history retention is limited.
- No forward-looking alignment is used; OI merge is latest `ts <= candle_close` only.

## Warnings
- BTC/USDT: Coverage ratio below 0.90; Binance OI history retention likely limited.
- ETH/USDT: Coverage ratio below 0.90; Binance OI history retention likely limited.
