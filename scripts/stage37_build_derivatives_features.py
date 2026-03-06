"""Stage-37.2 derivatives expansion with free Binance futures context series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.derived_store import load_derived_parquet, save_derived_parquet, write_meta_json
from buffmini.data.futures_extras import (
    align_long_short_ratio_to_ohlcv,
    align_taker_buy_sell_to_ohlcv,
    create_binance_futures_exchange,
    fetch_long_short_ratio_history_backfill,
    fetch_taker_buy_sell_history_backfill,
    long_short_ratio_quality_report,
    taker_buy_sell_quality_report,
)
from buffmini.data.store import build_data_store
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-37 derivatives features from free Binance futures data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=Path("data") / "derived")
    parser.add_argument("--last-days", type=int, default=365)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _coverage_years(frame: pd.DataFrame, *, ts_col: str = "timestamp") -> float:
    if frame.empty or ts_col not in frame.columns:
        return 0.0
    ts = pd.to_datetime(frame[ts_col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span_days = float((ts.iloc[-1] - ts.iloc[0]) / pd.Timedelta(days=1))
    return float(max(0.0, span_days / 365.25))


def _safe_load(kind: str, symbol: str, timeframe: str, derived_dir: Path) -> pd.DataFrame:
    try:
        return load_derived_parquet(kind=kind, symbol=symbol, timeframe=timeframe, data_dir=derived_dir)
    except FileNotFoundError:
        return pd.DataFrame()


def _csv_tokens(value: str, default: list[str]) -> list[str]:
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    return out or list(default)


def _report_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-37 Derivatives Expansion Report",
        "",
        "## Availability",
        f"- funding available: `{payload.get('funding_available', False)}`",
        f"- taker buy/sell available: `{payload.get('taker_available', False)}`",
        f"- long/short ratio available: `{payload.get('long_short_available', False)}`",
        f"- OI short-only mode enabled: `{payload.get('oi_short_only_mode', False)}`",
        "",
        "## Coverage by Symbol/Family",
        "| symbol | family | source | sample_count | start_ts | end_ts | coverage_years |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {symbol} | {family} | {source} | {samples} | {start} | {end} | {years:.4f} |".format(
                symbol=str(row.get("symbol", "")),
                family=str(row.get("family", "")),
                source=str(row.get("source", "")),
                samples=int(row.get("sample_count", 0)),
                start=str(row.get("start_ts", None)),
                end=str(row.get("end_ts", None)),
                years=float(row.get("coverage_years", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Funding is treated as long-history core context.",
            "- OI remains short-horizon only and is not mandatory for long-history training.",
            "- Taker and long/short families use free public Binance futures endpoints.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    fx_cfg = dict(data_cfg.get("futures_extras", {})) if isinstance(data_cfg, dict) else {}
    symbols = _csv_tokens(args.symbols, [str(v) for v in fx_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])])
    timeframe = str(fx_cfg.get("timeframe", "1h"))
    if timeframe != "1h":
        raise ValueError("Stage-37 derivatives expansion currently supports timeframe=1h")

    store = build_data_store(
        backend=str(data_cfg.get("backend", "parquet")),
        data_dir=args.data_dir,
        base_timeframe=str(cfg.get("universe", {}).get("base_timeframe") or timeframe),
        resample_source=str(data_cfg.get("resample_source", "direct")),
        derived_dir=args.derived_dir,
        partial_last_bucket=bool(data_cfg.get("partial_last_bucket", False)),
    )
    exchange = create_binance_futures_exchange()

    rows: list[dict[str, Any]] = []
    now_utc = utc_now_compact()
    for symbol in symbols:
        ohlcv = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if ohlcv.empty:
            continue
        start_ms = int(ohlcv["timestamp"].iloc[0].timestamp() * 1000)
        end_ms = int(ohlcv["timestamp"].iloc[-1].timestamp() * 1000)
        scoped_start_ms = max(start_ms, end_ms - int(max(1, args.last_days)) * 24 * 3600 * 1000)

        taker_raw, taker_info = fetch_taker_buy_sell_history_backfill(
            exchange=exchange,
            symbol=symbol,
            start_ms=scoped_start_ms,
            end_ms=end_ms,
            timeframe=timeframe,
            limit=500,
            max_retries=3,
            retry_backoff_sec=0.8,
            sleep_between_chunks_sec=0.0,
        )
        taker_aligned = align_taker_buy_sell_to_ohlcv(ohlcv=ohlcv, taker=taker_raw, timeframe=timeframe)
        save_derived_parquet(
            frame=taker_aligned,
            kind="taker_buy_sell",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        taker_q = taker_buy_sell_quality_report(taker_raw)
        write_meta_json(
            kind="taker_buy_sell",
            symbol=symbol,
            timeframe=timeframe,
            payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "kind": "taker_buy_sell",
                "source": "binance_public",
                "requests_count": int(taker_info.get("requests_count", 0)),
                "stop_reason": str(taker_info.get("stop_reason", "unknown")),
                "warnings": [str(v) for v in taker_info.get("warnings", [])],
                "quality": taker_q,
                "fetched_at_utc": now_utc,
            },
            data_dir=args.derived_dir,
        )

        ls_raw, ls_info = fetch_long_short_ratio_history_backfill(
            exchange=exchange,
            symbol=symbol,
            start_ms=scoped_start_ms,
            end_ms=end_ms,
            timeframe=timeframe,
            limit=500,
            max_retries=3,
            retry_backoff_sec=0.8,
            sleep_between_chunks_sec=0.0,
        )
        ls_aligned = align_long_short_ratio_to_ohlcv(ohlcv=ohlcv, long_short=ls_raw, timeframe=timeframe)
        save_derived_parquet(
            frame=ls_aligned,
            kind="long_short_ratio",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        ls_q = long_short_ratio_quality_report(ls_raw)
        write_meta_json(
            kind="long_short_ratio",
            symbol=symbol,
            timeframe=timeframe,
            payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "kind": "long_short_ratio",
                "source": "binance_public",
                "requests_count": int(ls_info.get("requests_count", 0)),
                "stop_reason": str(ls_info.get("stop_reason", "unknown")),
                "warnings": [str(v) for v in ls_info.get("warnings", [])],
                "quality": ls_q,
                "fetched_at_utc": now_utc,
            },
            data_dir=args.derived_dir,
        )

        for family, kind in (
            ("funding", "funding"),
            ("open_interest", "open_interest"),
            ("taker_buy_sell", "taker_buy_sell"),
            ("long_short_ratio", "long_short_ratio"),
        ):
            frame = _safe_load(kind=kind, symbol=symbol, timeframe=timeframe, derived_dir=args.derived_dir)
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna() if not frame.empty and "timestamp" in frame.columns else pd.Series(dtype="datetime64[ns, UTC]")
            rows.append(
                {
                    "symbol": symbol,
                    "family": family,
                    "source": "binance_public_futures",
                    "sample_count": int(frame.shape[0]),
                    "start_ts": ts.iloc[0].isoformat() if not ts.empty else None,
                    "end_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
                    "coverage_years": float(_coverage_years(frame)),
                }
            )

    funding_available = any(row["family"] == "funding" and int(row["sample_count"]) > 0 for row in rows)
    taker_available = any(row["family"] == "taker_buy_sell" and int(row["sample_count"]) > 0 for row in rows)
    long_short_available = any(row["family"] == "long_short_ratio" and int(row["sample_count"]) > 0 for row in rows)
    oi_short_only = bool(fx_cfg.get("open_interest", {}).get("short_horizon_only", False)) if isinstance(fx_cfg.get("open_interest", {}), dict) else False

    payload = {
        "stage": "37.2",
        "symbols": symbols,
        "timeframe": timeframe,
        "funding_available": bool(funding_available),
        "taker_available": bool(taker_available),
        "long_short_available": bool(long_short_available),
        "oi_short_only_mode": bool(oi_short_only),
        "rows": rows,
    }
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage37_derivatives_expansion_report.md"
    report_json = docs_dir / "stage37_derivatives_expansion_summary.json"
    report_md.write_text(_report_md(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")
    print(f"funding_available: {funding_available}")
    print(f"taker_available: {taker_available}")
    print(f"long_short_available: {long_short_available}")


if __name__ == "__main__":
    main()
