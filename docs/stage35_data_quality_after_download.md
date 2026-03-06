# Stage-35 Data Quality After Download

## Coverage Gate
- BTC/USDT funding_rates: years=0.010951, required=2.0, pass=False
- BTC/USDT open_interest: years=0.000000, required=2.0, pass=False
- ETH/USDT funding_rates: years=0.000000, required=2.0, pass=False
- ETH/USDT open_interest: years=0.000000, required=2.0, pass=False

## Per Endpoint Quality
- BTC/USDT funding_rates: samples=3, duplicates=0, monotonic=True, gaps=2, max_gap_hours=48.0, range=2026-01-01T00:00:00+00:00..2026-01-05T00:00:00+00:00
- BTC/USDT open_interest: samples=0, duplicates=0, monotonic=True, gaps=0, max_gap_hours=0.0, range=None..None
- ETH/USDT funding_rates: samples=0, duplicates=0, monotonic=True, gaps=0, max_gap_hours=0.0, range=None..None
- ETH/USDT open_interest: samples=0, duplicates=0, monotonic=True, gaps=0, max_gap_hours=0.0, range=None..None

## Alignment/Leakage
- BTC/USDT: bars=2103760, funding_non_null=2163, oi_non_null=0, funding_missing_ratio=0.998972, oi_missing_ratio=1.000000, leakage={'funding_rates_future_matches': 0, 'open_interest_future_matches': 0}
- ETH/USDT: bars=2103760, funding_non_null=0, oi_non_null=0, funding_missing_ratio=1.000000, oi_missing_ratio=1.000000, leakage={'funding_rates_future_matches': 0, 'open_interest_future_matches': 0}

## Derived Timeframe Integrity
- BTC/USDT 15m: file_exists=True, rows=140249, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- BTC/USDT 1h: file_exists=True, rows=35061, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- BTC/USDT 4h: file_exists=True, rows=8764, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- BTC/USDT 1d: file_exists=True, rows=1459, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- ETH/USDT 15m: file_exists=True, rows=140249, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- ETH/USDT 1h: file_exists=True, rows=35061, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- ETH/USDT 4h: file_exists=True, rows=8764, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
- ETH/USDT 1d: file_exists=True, rows=1459, duplicate_ts=0, monotonic_ts=True, integrity_ok=True
