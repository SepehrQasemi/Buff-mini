# Stage-35.4 CoinAPI Extras Feature Alignment

## Module
- `src/buffmini/features/extras_align.py`

## Alignment Method
- Uses backward `merge_asof` per endpoint (`funding_rates`, `open_interest`, `liquidations`).
- Only past values are allowed onto each OHLCV bar.
- No forward fill from future timestamps.

## Output Columns
- `funding_rate`
- `oi`
- `liq_buy`
- `liq_sell`
- `liq_notional`

## Determinism / Leakage Controls
- Inputs are sorted by `timestamp` / `ts` and deduplicated deterministically.
- Max staleness caps are enforced per endpoint.
- When stale or missing, aligned values are set to `NaN`.
- Tests include a synthetic future spike and verify no backward leakage.

## Pipeline Integration
- `calculate_features()` now supports:
  - `config.features.extras.enabled`
  - `config.features.extras.sources = ["coinapi"]`
- Defaults keep extras disabled, so classic behavior remains unchanged.

