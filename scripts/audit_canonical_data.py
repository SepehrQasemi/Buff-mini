"""Stage-26.9.2 canonical timeframe integrity and determinism audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, RAW_DATA_DIR
from buffmini.data.canonical_raw import file_sha256, prepare_frame, raw_path, symbol_safe
from buffmini.data.loader import standardize_ohlcv_frame, validate_ohlcv_frame
from buffmini.data.resample import resample_monthly_ohlcv, resample_ohlcv
from buffmini.data.timeframe_files import timeframe_to_file_token
from buffmini.utils.hashing import stable_hash

DEFAULT_CANONICAL_TFS = "1m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical OHLCV data integrity")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--base", type=str, default="1m")
    parser.add_argument("--timeframes", type=str, default=DEFAULT_CANONICAL_TFS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--drop-incomplete-last", action="store_true", default=True)
    return parser.parse_args()


def _parse_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def canonical_path(*, canonical_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    token = timeframe_to_file_token(str(timeframe))
    return Path(canonical_dir) / str(exchange).strip().lower() / symbol_safe(symbol) / f"{token}.parquet"


def canonical_meta_path(*, canonical_dir: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return canonical_path(canonical_dir=canonical_dir, exchange=exchange, symbol=symbol, timeframe=timeframe).with_suffix(
        ".meta.json"
    )


def _resample_expected(base_frame: pd.DataFrame, *, base_tf: str, target_tf: str, drop_incomplete_last: bool) -> pd.DataFrame:
    if str(target_tf) == str(base_tf):
        return base_frame.copy()
    if str(target_tf) == "1M":
        return resample_monthly_ohlcv(base_frame, partial_last_bucket=not bool(drop_incomplete_last))
    return resample_ohlcv(
        base_frame,
        target_timeframe=str(target_tf),
        base_timeframe=str(base_tf),
        partial_last_bucket=not bool(drop_incomplete_last),
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    core = standardize_ohlcv_frame(frame)
    payload = core.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list")
    return stable_hash(payload, length=24)


def _audit_one(
    *,
    symbol: str,
    exchange: str,
    base_tf: str,
    target_tf: str,
    raw_dir: Path,
    canonical_dir: Path,
    drop_incomplete_last: bool,
) -> dict[str, Any]:
    c_path = canonical_path(canonical_dir=canonical_dir, exchange=exchange, symbol=symbol, timeframe=target_tf)
    c_meta_path = canonical_meta_path(canonical_dir=canonical_dir, exchange=exchange, symbol=symbol, timeframe=target_tf)
    raw_file = raw_path(data_dir=raw_dir, exchange=exchange, symbol=symbol, timeframe=base_tf)

    if not c_path.exists() or not c_meta_path.exists() or not raw_file.exists():
        return {
            "symbol": str(symbol),
            "timeframe": str(target_tf),
            "exists": False,
            "path": str(c_path),
            "meta_path": str(c_meta_path),
            "raw_path": str(raw_file),
            "integrity_pass": False,
            "reason": "missing_file",
        }

    frame = standardize_ohlcv_frame(pd.read_parquet(c_path))
    validate_ohlcv_frame(frame)
    meta = json.loads(c_meta_path.read_text(encoding="utf-8"))

    base_frame = prepare_frame(pd.read_parquet(raw_file))
    expected = _resample_expected(base_frame, base_tf=base_tf, target_tf=target_tf, drop_incomplete_last=drop_incomplete_last)
    expected = standardize_ohlcv_frame(expected)
    validate_ohlcv_frame(expected)

    actual_hash = _frame_hash(frame)
    expected_hash = _frame_hash(expected)
    sha_match = str(meta.get("sha256", "")) == file_sha256(c_path)

    ratio_observed = None
    if int(frame.shape[0]) > 0 and int(base_frame.shape[0]) > 0:
        ratio_observed = float(base_frame.shape[0]) / float(frame.shape[0])

    return {
        "symbol": str(symbol),
        "timeframe": str(target_tf),
        "exists": True,
        "path": str(c_path),
        "meta_path": str(c_meta_path),
        "rows": int(frame.shape[0]),
        "start_ts": pd.to_datetime(frame["timestamp"], utc=True).iloc[0].isoformat() if not frame.empty else None,
        "end_ts": pd.to_datetime(frame["timestamp"], utc=True).iloc[-1].isoformat() if not frame.empty else None,
        "monotonic": bool(pd.to_datetime(frame["timestamp"], utc=True).is_monotonic_increasing),
        "duplicates": int(pd.to_datetime(frame["timestamp"], utc=True).duplicated().sum()),
        "sha_match": bool(sha_match),
        "meta_fields_ok": all(
            key in meta
            for key in ("timeframe", "source_timeframe", "start_ts", "end_ts", "candle_count", "sha256", "generated_at", "generator_version")
        ),
        "idempotent_rebuild_match": bool(actual_hash == expected_hash),
        "actual_data_hash": str(actual_hash),
        "expected_data_hash": str(expected_hash),
        "ratio_observed": ratio_observed,
        "integrity_pass": bool(
            (int(pd.to_datetime(frame["timestamp"], utc=True).duplicated().sum()) == 0)
            and bool(pd.to_datetime(frame["timestamp"], utc=True).is_monotonic_increasing)
            and bool(sha_match)
            and bool(actual_hash == expected_hash)
        ),
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-26.9 Canonical Data Audit",
        "",
        f"- integrity_pass: `{bool(payload.get('integrity_pass', False))}`",
        "",
        "| symbol | tf | rows | monotonic | dup | sha_match | rebuild_match | pass |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {int(row.get('rows',0))} | {bool(row.get('monotonic',False))} | {int(row.get('duplicates',0))} | {bool(row.get('sha_match',False))} | {bool(row.get('idempotent_rebuild_match',False))} | {bool(row.get('integrity_pass',False))} |"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    symbols = _parse_list(args.symbols)
    timeframes = _parse_list(args.timeframes)
    rows = []
    for symbol in symbols:
        for tf in timeframes:
            rows.append(
                _audit_one(
                    symbol=symbol,
                    exchange=str(args.exchange),
                    base_tf=str(args.base),
                    target_tf=tf,
                    raw_dir=Path(args.raw_dir),
                    canonical_dir=Path(args.canonical_dir),
                    drop_incomplete_last=bool(args.drop_incomplete_last),
                )
            )

    integrity_pass = all(bool(row.get("integrity_pass", False)) for row in rows)
    payload = {
        "stage": "26.9.2",
        "symbols": symbols,
        "timeframes": timeframes,
        "rows": rows,
        "integrity_pass": bool(integrity_pass),
    }
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "stage26_9_canonical_audit.md"
    json_path = docs_dir / "stage26_9_canonical_audit.json"
    md_path.write_text(_render_md(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"audit_md: {md_path}")
    print(f"audit_json: {json_path}")
    raise SystemExit(0 if integrity_pass else 2)


if __name__ == "__main__":
    main()
