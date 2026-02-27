"""Trade map plotting helper for Results Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.storage import load_parquet


def load_trade_artifact(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a trade-like CSV artifact from run folder if available."""

    warnings: list[str] = []
    candidates = sorted(run_dir.glob("trades*.csv"))
    if not candidates:
        fallback = run_dir / "orders.csv"
        if fallback.exists():
            candidates = [fallback]
    if not candidates:
        warnings.append("No trades/orders artifact found for trade map.")
        return pd.DataFrame(), warnings
    path = candidates[0]
    try:
        frame = pd.read_csv(path)
        return frame, warnings
    except Exception as exc:
        warnings.append(f"Failed to parse {path.name}: {exc}")
        return pd.DataFrame(), warnings


def plot_trade_map(
    trade_frame: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    data_dir: Path = RAW_DATA_DIR,
) -> tuple[plt.Figure | None, list[str]]:
    """Plot local price series and entry/exit markers for one symbol."""

    warnings: list[str] = []
    if trade_frame is None or trade_frame.empty:
        warnings.append("Trade frame is empty.")
        return None, warnings
    symbol_frame = trade_frame.copy()
    if "symbol" in symbol_frame.columns:
        symbol_frame = symbol_frame[symbol_frame["symbol"] == symbol]
    if symbol_frame.empty:
        warnings.append(f"No trade rows for symbol {symbol}.")
        return None, warnings

    timestamp_col = _pick_col(symbol_frame.columns, ["entry_time", "entry_ts", "ts", "timestamp"])
    if timestamp_col is None:
        warnings.append("No timestamp column found in trade artifact.")
        return None, warnings
    symbol_frame[timestamp_col] = pd.to_datetime(symbol_frame[timestamp_col], utc=True, errors="coerce")
    symbol_frame = symbol_frame.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if symbol_frame.empty:
        warnings.append("Trade timestamps are not parseable.")
        return None, warnings

    start = symbol_frame[timestamp_col].min() - pd.Timedelta(hours=12)
    end = symbol_frame[timestamp_col].max() + pd.Timedelta(hours=12)
    try:
        price = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    except Exception as exc:
        warnings.append(f"Price series unavailable for {symbol}: {exc}")
        return None, warnings
    price = price[(price["timestamp"] >= start) & (price["timestamp"] <= end)].copy()
    if price.empty:
        warnings.append("No overlapping price bars for trade map window.")
        return None, warnings

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price["timestamp"], price["close"], label="close", linewidth=1.2)

    for row in symbol_frame.to_dict(orient="records"):
        ts = row[timestamp_col]
        marker = "^" if int(row.get("direction", 1) or 1) >= 0 else "v"
        color = "green" if marker == "^" else "red"
        y = _nearest_close(price, ts)
        if y is None:
            continue
        ax.scatter([ts], [y], marker=marker, color=color, s=30)

    ax.set_title(f"Trade Map - {symbol}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, warnings


def _nearest_close(price: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if price.empty:
        return None
    idx = (price["timestamp"] - ts).abs().idxmin()
    return float(price.loc[idx, "close"])


def _pick_col(columns: Any, candidates: list[str]) -> str | None:
    existing = set(columns)
    for item in candidates:
        if item in existing:
            return item
    return None

