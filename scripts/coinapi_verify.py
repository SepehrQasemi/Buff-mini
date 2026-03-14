"""Single-request CoinAPI auth verification without exposing secret values."""

from __future__ import annotations

import re
from pathlib import Path

from buffmini.data.coinapi.client import CoinAPIClient, CoinAPIRequestError, resolve_coinapi_key_status
from buffmini.data.coinapi.secrets import resolve_coinapi_key


def _extract_http_status(message: str) -> int:
    match = re.search(r"\b([1-5][0-9]{2})\b", str(message or ""))
    if match is None:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _safe_reason(error: str) -> str:
    text = str(error or "").strip()
    if not text:
        return "verify_failed"
    if "401" in text:
        return "unauthorized"
    if "403" in text:
        return "forbidden"
    if "429" in text:
        return "rate_limited"
    if len(text) > 120:
        return text[:120]
    return text


def main() -> None:
    present, source = resolve_coinapi_key_status(repo_root=Path.cwd())
    if not present:
        print("ok=false http_status=0 key_source=MISSING reason=key_missing")
        raise SystemExit(1)

    key = str(resolve_coinapi_key(repo_root=Path.cwd()) or "").strip()
    if not key:
        print(f"ok=false http_status=0 key_source={source} reason=key_unreadable")
        raise SystemExit(1)

    try:
        client = CoinAPIClient(
            key,
            sleep_ms=0,
            max_total_requests=1,
            max_retries=0,
            repo_root=Path.cwd(),
        )
        _, meta = client.request_json(
            "/v1/exchanges",
            endpoint_name="verify_auth",
            symbol="verify",
        )
        print(f"ok=true http_status={int(meta.status_code)} key_source={source} reason=ok")
        return
    except CoinAPIRequestError as exc:
        status = _extract_http_status(str(exc))
        print(f"ok=false http_status={status} key_source={source} reason={_safe_reason(str(exc))}")
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        status = _extract_http_status(str(exc))
        print(f"ok=false http_status={status} key_source={source} reason={_safe_reason(str(exc))}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

