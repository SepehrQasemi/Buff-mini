# Stage-35 Real Download Report

## Status
- stage: `BLOCKED`
- reason: `MISSING_COINAPI_KEY`
- network download attempted: `false`
- stop point: `Step 1 (environment validation)`

## Environment Check
- `COINAPI_KEY` present: `NO`
- key value printed/logged: `NO`

## Commands Executed
1. `python -m compileall src scripts tests launch_app.py`
2. `python -m pytest -q`
3. `python -c "import os,sys; sys.exit(0 if os.environ.get('COINAPI_KEY','').strip() else 1)"`

## Gate Results
- compileall: `PASS`
- pytest: `457 passed`

## Why Download Did Not Start
- The task requires CoinAPI download scripts to use `COINAPI_KEY` from environment.
- The variable is currently unset in this shell/session.
- To avoid unsafe retries and to respect the no-secret/no-bruteforce policy, execution stopped before any API call.

## Next Action (Required)
Set the key in the active shell session, then rerun:

```powershell
$env:COINAPI_KEY="***your_key***"
python scripts/run_stage35.py --dry-run --seed 42
python scripts/update_coinapi_extras.py --plan --seed 42 --symbols BTC/USDT,ETH/USDT --years 4
python scripts/update_coinapi_extras.py --download --seed 42 --symbols BTC/USDT,ETH/USDT --years 4 --endpoints funding,oi --budget-requests 1500
```

After successful coverage (`>=2.0y` for BTC+ETH on funding+OI), run:

```powershell
python scripts/run_stage35.py --seed 42
```

## Notes
- Repo had pre-existing tracked doc modifications unrelated to this task; they were not altered here.
- No raw data files were staged or committed.

