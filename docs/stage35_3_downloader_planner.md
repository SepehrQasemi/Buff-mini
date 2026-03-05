# Stage-35.3 Downloader Planner

## Script
- `scripts/update_coinapi_extras.py`

## What It Does
- Builds a deterministic incremental backfill plan by `(endpoint, symbol, time-slice)`.
- Enforces a hard request cap before execution.
- Writes plan artifact to:
  - `runs/<run_id>/stage35/coinapi_plan.json`
- Supports `--dry-run` plan-only mode.

## Budget Controls
- Config/request cap: `coinapi.max_total_requests`
- CLI cap override: `--max-requests` (still clamped by config cap)
- Slice size: `--increment-days`
- Range: `--start/--end` or `--last-days`

## Storage Behavior
- Raw rows (optional): `data/coinapi/raw/coinapi/<symbol>/<endpoint>/<yyyy>/<mm>/<dd>.jsonl.gz`
- Canonical normalized output: `data/coinapi/canonical/<symbol>/<endpoint>.parquet`
- Usage ledger: `data/coinapi/meta/usage_ledger.jsonl`

## Safety
- `--offline` refuses execution when `coinapi.enabled=true`.
- `COINAPI_KEY` is required for non-dry runs.
- Raw storage auto-disables once cumulative raw size exceeds `2GB`.

