# Stage-35.1 CoinAPI Client

## Environment
- `COINAPI_KEY` is required when CoinAPI ingestion is enabled.
- `COINAPI_BASE_URL` is optional and defaults to `https://rest.coinapi.io`.

## Client Safety
- API key is only sent in header `X-CoinAPI-Key`.
- Ledger/log output stores only a masked key suffix (`***1234`).
- Request query params are sanitized to remove token-like keys.
- Requests are hard-capped by `max_total_requests`.

## Usage Ledger Fields
- `ts_utc`, `endpoint_name`, `http_method`, `request_url_path`
- `query_params`, `symbol`, `time_start`, `time_end`
- `status_code`, `response_bytes`
- `header_signals` (quota/rate/credit headers when present)
- `retry_count`, `elapsed_ms`, `error_message`, `plan_id`

## Retry Behavior
- Retries are bounded and only for transient failures (`429`, `5xx`, timeout/network).
- Exponential backoff is deterministic from configured `sleep_ms`.
- Every request attempt outcome is recorded to ledger.

