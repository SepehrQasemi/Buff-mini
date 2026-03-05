"""CoinAPI HTTP client with retry/backoff and usage-ledger instrumentation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable
from urllib import error, parse, request

from .secrets import resolve_coinapi_key
from .usage import CoinAPIUsageLedger, parse_coinapi_header_signals


class CoinAPIRequestError(RuntimeError):
    """Raised when CoinAPI request fails after retries."""


@dataclass(frozen=True, slots=True)
class CoinAPIResponseMeta:
    status_code: int
    response_bytes: int
    headers: dict[str, Any]
    header_signals: dict[str, Any]
    elapsed_ms: int
    retry_count: int
    endpoint_path: str
    endpoint_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": int(self.status_code),
            "response_bytes": int(self.response_bytes),
            "headers": dict(self.headers),
            "header_signals": dict(self.header_signals),
            "elapsed_ms": int(self.elapsed_ms),
            "retry_count": int(self.retry_count),
            "endpoint_path": str(self.endpoint_path),
            "endpoint_name": str(self.endpoint_name),
        }


TransportFn = Callable[[str, dict[str, str], int], tuple[int, bytes, dict[str, Any]]]


def _default_transport(url: str, headers: dict[str, str], timeout_sec: int) -> tuple[int, bytes, dict[str, Any]]:
    req = request.Request(url=url, method="GET", headers=headers)
    with request.urlopen(req, timeout=int(timeout_sec)) as response:
        body = response.read()
        status = int(response.status)
        hdrs = {str(k): str(v) for k, v in response.headers.items()}
    return status, body, hdrs


class CoinAPIClient:
    """Simple bounded client for CoinAPI REST requests."""

    def __init__(
        self,
        key: str | None = None,
        *,
        base_url: str = "https://rest.coinapi.io",
        sleep_ms: int = 120,
        max_total_requests: int = 2000,
        timeout_sec: int = 30,
        max_retries: int = 3,
        ledger: CoinAPIUsageLedger | None = None,
        transport: TransportFn | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        token = str(key or "").strip()
        if not token:
            root = Path(repo_root).resolve() if repo_root is not None else None
            token = str(resolve_coinapi_key(repo_root=root) or "").strip()
        if not token:
            raise ValueError("CoinAPI key is required (COINAPI_KEY or secrets/coinapi_key.txt)")
        self._key = token
        self.base_url = str(base_url).rstrip("/")
        self.sleep_ms = max(0, int(sleep_ms))
        self.max_total_requests = max(1, int(max_total_requests))
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_retries = max(0, int(max_retries))
        self.ledger = ledger
        self._transport = transport or _default_transport
        self._request_count = 0

    @property
    def request_count(self) -> int:
        return int(self._request_count)

    @property
    def masked_key(self) -> str:
        tail = self._key[-4:] if len(self._key) >= 4 else self._key
        return f"***{tail}"

    def request_json(
        self,
        endpoint_path: str,
        *,
        params: dict[str, Any] | None = None,
        endpoint_name: str | None = None,
        symbol: str | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        plan_id: str | None = None,
    ) -> tuple[Any, CoinAPIResponseMeta]:
        path = "/" + str(endpoint_path).lstrip("/")
        query = parse.urlencode(_safe_query_params(params or {}), doseq=True)
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        ep_name = str(endpoint_name or path.strip("/").replace("/", "_"))

        if self._request_count >= self.max_total_requests:
            message = "max_total_requests_exceeded"
            self._write_ledger(
                endpoint_name=ep_name,
                endpoint_path=path,
                query_params=params or {},
                symbol=symbol,
                time_start=time_start,
                time_end=time_end,
                status_code=0,
                response_bytes=0,
                header_signals={},
                retry_count=0,
                elapsed_ms=0,
                error_message=message,
                plan_id=plan_id,
            )
            raise CoinAPIRequestError(message)

        headers = {
            "Accept": "application/json",
            "X-CoinAPI-Key": self._key,
        }

        attempt = 0
        last_exc: Exception | None = None
        start = time.perf_counter()
        status_code = 0
        response_bytes = 0
        response_headers: dict[str, Any] = {}
        parsed_payload: Any = None

        while attempt <= self.max_retries:
            self._request_count += 1
            try:
                status_code, body, response_headers = self._transport(url, headers, self.timeout_sec)
                response_bytes = int(len(body))
                if status_code == 429 or status_code >= 500:
                    raise CoinAPIRequestError(f"http_{status_code}")
                if status_code < 200 or status_code >= 300:
                    text = body.decode("utf-8", errors="replace")
                    raise CoinAPIRequestError(f"http_{status_code}:{text[:256]}")
                parsed_payload = json.loads(body.decode("utf-8"))
                break
            except error.HTTPError as exc:
                status_code = int(getattr(exc, "code", 0) or 0)
                body = exc.read() if hasattr(exc, "read") else b""
                response_bytes = int(len(body))
                response_headers = {str(k): str(v) for k, v in dict(getattr(exc, "headers", {}) or {}).items()}
                last_exc = exc
                if status_code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    break
            except (error.URLError, TimeoutError, CoinAPIRequestError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
            attempt += 1
            if self.sleep_ms > 0:
                time.sleep((self.sleep_ms / 1000.0) * float(2**attempt))

        elapsed_ms = int(round((time.perf_counter() - start) * 1000.0))
        header_signals = parse_coinapi_header_signals(response_headers)
        retry_count = int(attempt)

        if parsed_payload is None:
            message = str(last_exc) if last_exc is not None else "request_failed"
            self._write_ledger(
                endpoint_name=ep_name,
                endpoint_path=path,
                query_params=params or {},
                symbol=symbol,
                time_start=time_start,
                time_end=time_end,
                status_code=int(status_code),
                response_bytes=int(response_bytes),
                header_signals=header_signals,
                retry_count=retry_count,
                elapsed_ms=elapsed_ms,
                error_message=message,
                plan_id=plan_id,
            )
            raise CoinAPIRequestError(message)

        self._write_ledger(
            endpoint_name=ep_name,
            endpoint_path=path,
            query_params=params or {},
            symbol=symbol,
            time_start=time_start,
            time_end=time_end,
            status_code=int(status_code),
            response_bytes=int(response_bytes),
            header_signals=header_signals,
            retry_count=retry_count,
            elapsed_ms=elapsed_ms,
            error_message="",
            plan_id=plan_id,
        )
        meta = CoinAPIResponseMeta(
            status_code=int(status_code),
            response_bytes=int(response_bytes),
            headers=dict(MappingProxyType(response_headers)),
            header_signals=dict(header_signals),
            elapsed_ms=int(elapsed_ms),
            retry_count=int(retry_count),
            endpoint_path=path,
            endpoint_name=ep_name,
        )
        return parsed_payload, meta

    def _write_ledger(
        self,
        *,
        endpoint_name: str,
        endpoint_path: str,
        query_params: dict[str, Any],
        symbol: str | None,
        time_start: str | None,
        time_end: str | None,
        status_code: int,
        response_bytes: int,
        header_signals: dict[str, Any],
        retry_count: int,
        elapsed_ms: int,
        error_message: str,
        plan_id: str | None,
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "endpoint_name": str(endpoint_name),
                "http_method": "GET",
                "request_url_path": str(endpoint_path),
                "query_params": _safe_query_params(query_params),
                "symbol": str(symbol or ""),
                "time_start": str(time_start or ""),
                "time_end": str(time_end or ""),
                "status_code": int(status_code),
                "response_bytes": int(response_bytes),
                "header_signals": dict(header_signals),
                "retry_count": int(retry_count),
                "elapsed_ms": int(elapsed_ms),
                "error_message": str(error_message),
                "plan_id": str(plan_id or ""),
                "masked_api_key": self.masked_key,
            }
        )


def _safe_query_params(params: dict[str, Any]) -> dict[str, Any]:
    blocked = {"apikey", "api_key", "x_coinapi_key", "key", "token"}
    safe: dict[str, Any] = {}
    for key, value in params.items():
        k = str(key)
        if k.lower() in blocked:
            continue
        if isinstance(value, (list, tuple)):
            safe[k] = [str(v) for v in value]
        elif value is None:
            continue
        else:
            safe[k] = str(value)
    return safe
