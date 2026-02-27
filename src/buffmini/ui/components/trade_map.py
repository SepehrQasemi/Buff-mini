"""Trade map component using standardized ui_bundle artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.storage import load_parquet


_DIRECTION_FILTERS = {"both", "long", "short"}


def load_bundle_summary(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load ui_bundle summary for a pipeline run."""

    warnings: list[str] = []
    path = Path(run_dir) / "ui_bundle" / "summary_ui.json"
    if not path.exists():
        warnings.append(f"Missing {path}")
        return {}, warnings
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"Failed to parse summary_ui.json: {exc}")
        return {}, warnings
    return payload if isinstance(payload, dict) else {}, warnings


def load_bundle_trades(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load ui_bundle/trades.csv if available."""

    warnings: list[str] = []
    path = Path(run_dir) / "ui_bundle" / "trades.csv"
    if not path.exists():
        warnings.append("ui_bundle/trades.csv not found")
        return pd.DataFrame(), warnings
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Failed to read trades.csv: {exc}")
        return pd.DataFrame(), warnings
    return frame, warnings


def load_price_window(
    symbol: str,
    timeframe: str,
    start_ts: str | pd.Timestamp | None,
    end_ts: str | pd.Timestamp | None,
    data_dir: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, list[str]]:
    """Load local price series window from parquet cache."""

    warnings: list[str] = []
    try:
        frame = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    except Exception as exc:
        warnings.append(f"Price data missing for {symbol} {timeframe}: {exc}")
        return pd.DataFrame(), warnings

    if frame.empty:
        warnings.append(f"Price dataframe empty for {symbol} {timeframe}")
        return frame, warnings

    start = pd.to_datetime(start_ts, utc=True, errors="coerce") if start_ts is not None else pd.NaT
    end = pd.to_datetime(end_ts, utc=True, errors="coerce") if end_ts is not None else pd.NaT

    if pd.notna(start):
        frame = frame[frame["timestamp"] >= start]
    if pd.notna(end):
        frame = frame[frame["timestamp"] <= end]

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        warnings.append("No price bars in requested run window.")
    return frame, warnings


def build_trade_markers(
    trades: pd.DataFrame,
    symbol: str,
    direction_filter: str = "both",
    execution_mode: str = "net",
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize trades into marker rows for plotting."""

    warnings: list[str] = []
    if direction_filter not in _DIRECTION_FILTERS:
        warnings.append(f"Invalid direction filter `{direction_filter}`, using `both`.")
        direction_filter = "both"

    if trades is None or trades.empty:
        warnings.append("Trades dataframe is empty.")
        return pd.DataFrame(), warnings

    frame = trades.copy()
    ts_col = _pick_column(frame.columns, ["timestamp", "ts", "entry_time"])
    if ts_col is None:
        warnings.append("Trades missing timestamp column.")
        return pd.DataFrame(), warnings
    frame["timestamp"] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")

    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str) == symbol]
    else:
        frame["symbol"] = symbol

    if frame.empty:
        warnings.append(f"No trade rows for symbol {symbol}.")
        return pd.DataFrame(), warnings

    frame["direction"] = frame.apply(_row_direction, axis=1)
    frame["action"] = frame.get("action", "entry").astype(str)
    frame["notional_fraction_of_equity"] = pd.to_numeric(
        frame.get("notional_fraction_of_equity", 1.0), errors="coerce"
    ).fillna(1.0)

    frame = frame.dropna(subset=["timestamp"]).copy()

    if str(execution_mode).lower() == "net":
        grouped = (
            frame.groupby(["timestamp", "symbol"], as_index=False)
            .agg(
                net_direction=("direction", "sum"),
                gross_size=("notional_fraction_of_equity", "sum"),
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        grouped["direction"] = grouped["net_direction"].apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        grouped = grouped[grouped["direction"] != 0].copy()
        grouped["action"] = "entry"
        markers = grouped[["timestamp", "symbol", "direction", "action", "gross_size"]].copy()
    else:
        markers = frame[["timestamp", "symbol", "direction", "action", "notional_fraction_of_equity"]].copy()
        markers = markers.rename(columns={"notional_fraction_of_equity": "gross_size"})

    if direction_filter == "long":
        markers = markers[markers["direction"] > 0]
    elif direction_filter == "short":
        markers = markers[markers["direction"] < 0]

    markers = markers.sort_values("timestamp").reset_index(drop=True)
    if markers.empty:
        warnings.append("No markers after filtering.")
    return markers, warnings


def plot_trade_map(
    run_dir: Path,
    symbol: str,
    direction_filter: str = "both",
    data_dir: Path = RAW_DATA_DIR,
) -> tuple[plt.Figure | None, list[str], pd.DataFrame]:
    """Plot price and trade markers for a run using ui_bundle artifacts."""

    warnings: list[str] = []
    summary, summary_warnings = load_bundle_summary(run_dir)
    warnings.extend(summary_warnings)
    trades, trades_warnings = load_bundle_trades(run_dir)
    warnings.extend(trades_warnings)

    if not summary:
        return None, warnings, pd.DataFrame()

    execution_mode = str(summary.get("execution_mode", "net"))
    timeframe = str(summary.get("timeframe", "1h"))
    start_ts = summary.get("run_window_start_ts")
    end_ts = summary.get("run_window_end_ts")

    markers, marker_warnings = build_trade_markers(
        trades=trades,
        symbol=symbol,
        direction_filter=direction_filter,
        execution_mode=execution_mode,
    )
    warnings.extend(marker_warnings)

    if (start_ts is None or end_ts is None) and not markers.empty:
        start_ts = markers["timestamp"].min() - pd.Timedelta(hours=12)
        end_ts = markers["timestamp"].max() + pd.Timedelta(hours=12)

    price, price_warnings = load_price_window(
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        data_dir=data_dir,
    )
    warnings.extend(price_warnings)

    if price.empty:
        return None, warnings, markers

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(price["timestamp"], price["close"], linewidth=1.1, label="close")

    if not markers.empty:
        long_markers = markers[markers["direction"] > 0]
        short_markers = markers[markers["direction"] < 0]

        _scatter_markers(ax, price, long_markers, color="green", marker="^", label="long")
        _scatter_markers(ax, price, short_markers, color="red", marker="v", label="short")

    ax.set_title(f"Trade Map | {symbol} | mode={execution_mode}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, warnings, markers


def _scatter_markers(
    ax: Any,
    price: pd.DataFrame,
    markers: pd.DataFrame,
    color: str,
    marker: str,
    label: str,
) -> None:
    if markers.empty:
        return
    ys = []
    for ts in markers["timestamp"]:
        ys.append(_nearest_close(price, ts))
    ax.scatter(markers["timestamp"], ys, color=color, marker=marker, s=28, label=label)


def _nearest_close(price: pd.DataFrame, ts: pd.Timestamp) -> float:
    idx = (price["timestamp"] - ts).abs().idxmin()
    return float(price.loc[idx, "close"])


def _pick_column(columns: Any, candidates: list[str]) -> str | None:
    colset = set(columns)
    for item in candidates:
        if item in colset:
            return item
    return None


def _row_direction(row: pd.Series) -> int:
    if "direction" in row and pd.notna(row["direction"]):
        try:
            value = float(row["direction"])
            if value > 0:
                return 1
            if value < 0:
                return -1
        except Exception:
            pass

    side = str(row.get("side", "")).lower()
    if side == "long":
        return 1
    if side == "short":
        return -1
    return 1
