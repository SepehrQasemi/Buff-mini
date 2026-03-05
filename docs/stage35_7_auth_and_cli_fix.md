# Stage-35.7 Auth and CLI Fix

## Secret Resolution Order
1. `COINAPI_KEY` from environment
2. `secrets/coinapi_key.txt` (gitignored)
3. `secrets/coinapi_key.json` with `COINAPI_KEY` key (gitignored)
4. repo `.env` with `COINAPI_KEY=...` (gitignored)

## Key Doctor Commands
- status: `python scripts/coinapi_key_doctor.py --status`
- wipe local secret files: `python scripts/coinapi_key_doctor.py --wipe-old`
- write key interactively: `python scripts/coinapi_key_doctor.py --write`

## CLI Aliases Added
- `--download` => download action
- `--plan` => planning action
- `--verify` => auth probe (single lightweight request)
- `--years N` => backfill range alias
- `--budget-requests N` => alias for `--max-requests N`
- endpoint aliases: `funding` => `funding_rates`, `oi` => `open_interest`

## Missing Key Error
When key is unavailable, downloader exits with:
`COINAPI_KEY missing; use secrets/coinapi_key.txt (gitignored) or environment variable.`
