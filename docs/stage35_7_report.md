# Stage-35.7 Report

- status: `VERIFY_FAILED`
- key_source: `SECRETS_TXT`
- verify_ok: `False`
- plan_within_budget: `False`
- planned_requests: `0`
- selected_requests: `0`
- requests_made: `0`
- status_code_counts: `{'401': 3}`
- coverage_ok: `False`
- biggest_blocker: `CoinAPI auth verification failed`

## Coverage Years
- not available

## Next Actions
- `python scripts/coinapi_key_doctor.py --status`
- `python scripts/coinapi_key_doctor.py --wipe-old`
- `python scripts/coinapi_key_doctor.py --write`
- `python scripts/update_coinapi_extras.py --verify --config configs/local_coinapi.yaml --seed 42 --symbols BTC/USDT,ETH/USDT --endpoints funding,oi --years 4 --increment-days 7 --max-requests 1500`

## Command Evidence
### key_status stdout
```text
OK source=SECRETS_TXT

```
### key_status stderr
```text

```
### verify stdout
```text

```
### verify stderr
```text
Auth failed during verify endpoint=verify_auth error=HTTP Error 401: Unauthorized

```
### plan stdout
```text

```
### plan stderr
```text

```
### download stdout
```text

```
### download stderr
```text

```
