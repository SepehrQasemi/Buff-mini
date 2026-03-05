# Stage-35.2 Endpoint Schemas

## Implemented Adapters
- `funding_rates`
- `open_interest`
- `liquidations` (implemented, optional in default priority)

## Canonical Output Contract
All normalized endpoint frames use:
- `ts` (UTC timestamp)
- `symbol`
- endpoint value fields
- `source`
- `ingest_ts`

Endpoint value columns:
- funding: `funding_rate`
- open interest: `open_interest`
- liquidations: `liq_buy`, `liq_sell`, `liq_notional`

## Quality Rules
- Sort strictly ascending by `ts`.
- Drop duplicate timestamps (`keep='last'` deterministic rule).
- Reject non-finite value rows.
- Coverage diagnostics include:
  - `sample_count`
  - `start_ts`, `end_ts`
  - `missing_ratio`
  - `gaps_count`
  - `max_gap_minutes`

Gap thresholds:
- funding: `> 6h` reported using `8h` cadence baseline
- open interest: `> 1d`
- liquidations: `> 6h`

