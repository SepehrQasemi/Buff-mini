"""Leakage-safe alignment of CoinAPI extras onto OHLCV bars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def load_coinapi_endpoint_frame(
    *,
    symbol: str,
    endpoint: str,
    canonical_root: str | Path = Path("data") / "coinapi" / "canonical",
) -> pd.DataFrame:
    path = Path(canonical_root) / _safe_symbol(symbol) / f"{endpoint}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ts"])
    frame = pd.read_parquet(path)
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    return frame


def align_coinapi_extras_to_bars(
    bars: pd.DataFrame,
    *,
    funding: pd.DataFrame | None = None,
    open_interest: pd.DataFrame | None = None,
    liquidations: pd.DataFrame | None = None,
    max_staleness: dict[str, str] | None = None,
) -> pd.DataFrame:
    """As-of align extras to bars using backward-only matching (no future leak)."""

    if "timestamp" not in bars.columns:
        raise ValueError("bars must include timestamp")
    work = bars[["timestamp"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    result = work.copy()

    stale = {
        "funding_rates": pd.Timedelta((max_staleness or {}).get("funding_rates", "12h")),
        "open_interest": pd.Timedelta((max_staleness or {}).get("open_interest", "2d")),
        "liquidations": pd.Timedelta((max_staleness or {}).get("liquidations", "6h")),
    }

    def _merge(endpoint: str, frame: pd.DataFrame | None, value_cols: list[str], renames: dict[str, str] | None = None) -> None:
        nonlocal result
        if frame is None or frame.empty:
            for value_col in value_cols:
                col = renames.get(value_col, value_col) if renames else value_col
                result[col] = pd.NA
            return
        src = frame.copy()
        src["ts"] = pd.to_datetime(src["ts"], utc=True, errors="coerce")
        src = src.dropna(subset=["ts"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        cols = ["ts", *[c for c in value_cols if c in src.columns]]
        src = src[cols]
        merged = pd.merge_asof(result[["timestamp"]], src, left_on="timestamp", right_on="ts", direction="backward")
        age = merged["timestamp"] - merged["ts"]
        too_old = (age > stale[endpoint]) | merged["ts"].isna()
        for value_col in value_cols:
            out_col = renames.get(value_col, value_col) if renames else value_col
            series = merged[value_col] if value_col in merged.columns else pd.Series([pd.NA] * len(merged), index=merged.index)
            series = series.mask(too_old)
            result[out_col] = series

    _merge("funding_rates", funding, ["funding_rate"])
    _merge("open_interest", open_interest, ["open_interest"], renames={"open_interest": "oi"})
    _merge("liquidations", liquidations, ["liq_buy", "liq_sell", "liq_notional"])
    return result


def config_extras_enabled(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    features_cfg = config.get("features", {})
    if not isinstance(features_cfg, dict):
        return False
    extras_cfg = features_cfg.get("extras", {})
    if not isinstance(extras_cfg, dict):
        return False
    if not bool(extras_cfg.get("enabled", False)):
        return False
    sources = extras_cfg.get("sources", [])
    if not isinstance(sources, list):
        return False
    return "coinapi" in {str(v).strip().lower() for v in sources}

