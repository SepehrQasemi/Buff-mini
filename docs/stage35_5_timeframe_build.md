# Stage-35.5 Derived Timeframe Builder

## Policy
- Prefer `1m` as the canonical truth base whenever it exists and target TF is divisible.
- If `1m` is unavailable, use the smallest divisible available base.
- Reject non-divisible target timeframes explicitly.

## Deterministic Resampling
- UTC aligned boundaries (left-closed, left-labeled).
- Aggregation:
  - `open=first`
  - `high=max`
  - `low=min`
  - `close=last`
  - `volume=sum`
- Incomplete last bucket dropped by default.

## Integrity Checks
- `high >= max(open, close)` and `low <= min(open, close)` for every bar.
- Re-aggregated volume matches expected target bucket sums within tolerance.
- Tested across:
  - `5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1M`

