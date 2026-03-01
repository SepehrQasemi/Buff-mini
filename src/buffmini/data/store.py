"""Backend-ready market data store abstraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.cache import DerivedTimeframeCache, get_or_build_derived_ohlcv, ohlcv_data_hash
from buffmini.data.resample import resample_ohlcv, resample_settings_hash
from buffmini.data.storage import load_parquet, parquet_path, save_parquet
from buffmini.utils.hashing import stable_hash


class DataStore(Protocol):
    """Minimal market-data backend contract."""

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV rows for one symbol/timeframe."""

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Persist OHLCV rows for one symbol/timeframe."""

    def list_series(self) -> list[dict[str, str]]:
        """List known symbol/timeframe series."""

    def coverage(self, symbol: str, timeframe: str) -> dict[str, Any]:
        """Return basic availability metadata."""


class ParquetStore:
    """Parquet-backed store. This remains the source of truth."""

    def __init__(
        self,
        data_dir: str | Path = RAW_DATA_DIR,
        *,
        base_timeframe: str | None = None,
        resample_source: str = "direct",
        derived_dir: str | Path | None = None,
        partial_last_bucket: bool = False,
        config_hash: str | None = None,
        resolved_end_ts: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.base_timeframe = str(base_timeframe).strip().lower() if base_timeframe else None
        self.resample_source = str(resample_source).strip().lower()
        self.partial_last_bucket = bool(partial_last_bucket)
        self.config_hash = str(config_hash or "na")
        self.resolved_end_ts = str(resolved_end_ts or "")
        resolved_derived_dir = Path(derived_dir) if derived_dir is not None else (self.data_dir.parent / "derived")
        self.derived_cache = DerivedTimeframeCache(resolved_derived_dir)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        resolved_timeframe = str(timeframe).strip().lower()
        should_resample = (
            self.resample_source == "base"
            and bool(self.base_timeframe)
            and str(resolved_timeframe) != str(self.base_timeframe)
        )
        if should_resample:
            base_frame = load_parquet(symbol=symbol, timeframe=str(self.base_timeframe), data_dir=self.data_dir)
            source_hash = ohlcv_data_hash(base_frame)
            settings_hash = stable_hash(
                resample_settings_hash(
                    base_timeframe=str(self.base_timeframe),
                    target_timeframe=resolved_timeframe,
                    partial_last_bucket=self.partial_last_bucket,
                ),
                length=16,
            )
            frame, _ = get_or_build_derived_ohlcv(
                cache=self.derived_cache,
                symbol=symbol,
                base_tf=str(self.base_timeframe),
                target_tf=resolved_timeframe,
                start_ts=_stringify_ts(start),
                end_ts=_stringify_ts(end),
                resolved_end_ts=self.resolved_end_ts or _stringify_ts(end),
                data_hash=stable_hash(
                    {
                        "base_hash": source_hash,
                        "settings_hash": settings_hash,
                    },
                    length=16,
                ),
                config_hash=self.config_hash,
                builder=lambda: resample_ohlcv(
                    base_frame,
                    target_timeframe=resolved_timeframe,
                    base_timeframe=str(self.base_timeframe),
                    partial_last_bucket=self.partial_last_bucket,
                ),
            )
        else:
            frame = load_parquet(symbol=symbol, timeframe=resolved_timeframe, data_dir=self.data_dir)
        return _filter_frame(frame=frame, start=start, end=end)

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        save_parquet(frame=df, symbol=symbol, timeframe=timeframe, data_dir=self.data_dir)

    def list_series(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for path in sorted(self.data_dir.glob("*.parquet")):
            stem = path.stem
            if "_" not in stem:
                continue
            safe_symbol, timeframe = stem.rsplit("_", 1)
            items.append(
                {
                    "symbol": safe_symbol.replace("-", "/"),
                    "timeframe": timeframe,
                    "path": str(path),
                }
            )
        return items

    def coverage(self, symbol: str, timeframe: str) -> dict[str, Any]:
        resolved_timeframe = str(timeframe).strip().lower()
        should_resample = (
            self.resample_source == "base"
            and bool(self.base_timeframe)
            and str(resolved_timeframe) != str(self.base_timeframe)
        )
        path = (
            self.derived_cache.cache_path(symbol=symbol, timeframe=resolved_timeframe)
            if should_resample
            else parquet_path(symbol=symbol, timeframe=resolved_timeframe, data_dir=self.data_dir)
        )
        if not path.exists():
            return {
                "backend": "parquet",
                "symbol": symbol,
                "timeframe": resolved_timeframe,
                "exists": False,
                "rows": 0,
                "start": None,
                "end": None,
                "path": str(path),
            }

        frame = self.load_ohlcv(symbol=symbol, timeframe=resolved_timeframe)
        if frame.empty:
            return {
                "backend": "parquet",
                "symbol": symbol,
                "timeframe": resolved_timeframe,
                "exists": True,
                "rows": 0,
                "start": None,
                "end": None,
                "path": str(path),
            }

        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        return {
            "backend": "parquet",
            "symbol": symbol,
            "timeframe": resolved_timeframe,
            "exists": True,
            "rows": int(len(frame)),
            "start": timestamps.iloc[0].isoformat(),
            "end": timestamps.iloc[-1].isoformat(),
            "path": str(path),
        }


class DuckDBStore:
    """Optional in-process DuckDB cache backed by parquet source-of-truth."""

    def __init__(
        self,
        data_dir: str | Path = RAW_DATA_DIR,
        db_path: str | Path | None = None,
    ) -> None:
        self.parquet_store = ParquetStore(data_dir=data_dir)
        resolved_data_dir = Path(data_dir)
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else resolved_data_dir.parent / "cache" / "market.duckdb"
        )

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        con = self._connect()
        if con is None or not self._table_exists(con):
            return self.parquet_store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)

        query = (
            "SELECT timestamp, open, high, low, close, volume "
            "FROM ohlcv WHERE symbol = ? AND timeframe = ?"
        )
        params: list[Any] = [symbol, timeframe]
        if start is not None:
            query += " AND timestamp >= ?"
            params.append(_as_timestamp(start))
        if end is not None:
            query += " AND timestamp <= ?"
            params.append(_as_timestamp(end))
        query += " ORDER BY timestamp"

        frame = con.execute(query, params).df()
        con.close()
        if frame.empty:
            return self.parquet_store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start, end=end)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self.parquet_store.save_ohlcv(symbol=symbol, timeframe=timeframe, df=df)

        con = self._connect()
        if con is None:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                timeframe VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
            """
        )
        con.execute("DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?", [symbol, timeframe])
        write_df = df.copy()
        write_df["timestamp"] = pd.to_datetime(write_df["timestamp"], utc=True).dt.tz_localize(None)
        write_df["symbol"] = symbol
        write_df["timeframe"] = timeframe
        con.register("incoming_ohlcv", write_df[["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"]])
        con.execute("INSERT INTO ohlcv SELECT * FROM incoming_ohlcv")
        con.unregister("incoming_ohlcv")
        con.close()

    def list_series(self) -> list[dict[str, str]]:
        con = self._connect()
        if con is not None and self._table_exists(con):
            rows = con.execute(
                "SELECT DISTINCT symbol, timeframe FROM ohlcv ORDER BY symbol, timeframe"
            ).fetchall()
            con.close()
            if rows:
                return [
                    {"symbol": str(symbol), "timeframe": str(timeframe), "path": str(self.db_path)}
                    for symbol, timeframe in rows
                ]
        elif con is not None:
            con.close()
        return self.parquet_store.list_series()

    def coverage(self, symbol: str, timeframe: str) -> dict[str, Any]:
        con = self._connect()
        if con is not None and self._table_exists(con):
            row = con.execute(
                """
                SELECT COUNT(*) AS rows, MIN(timestamp) AS start_ts, MAX(timestamp) AS end_ts
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                """,
                [symbol, timeframe],
            ).fetchone()
            con.close()
            if row and int(row[0]) > 0:
                return {
                    "backend": "duckdb",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "exists": True,
                    "rows": int(row[0]),
                    "start": pd.Timestamp(row[1], tz="UTC").isoformat() if row[1] is not None else None,
                    "end": pd.Timestamp(row[2], tz="UTC").isoformat() if row[2] is not None else None,
                    "path": str(self.db_path),
                }
        elif con is not None:
            con.close()
        return self.parquet_store.coverage(symbol=symbol, timeframe=timeframe)

    def _connect(self) -> Any | None:
        try:
            import duckdb  # type: ignore
        except ImportError:
            return None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.db_path))

    @staticmethod
    def _table_exists(con: Any) -> bool:
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'ohlcv'"
        ).fetchone()
        return bool(row and int(row[0]) > 0)


def build_data_store(
    backend: str = "parquet",
    data_dir: str | Path = RAW_DATA_DIR,
    db_path: str | Path | None = None,
    *,
    base_timeframe: str | None = None,
    resample_source: str = "direct",
    derived_dir: str | Path | None = None,
    partial_last_bucket: bool = False,
    config_hash: str | None = None,
    resolved_end_ts: str | None = None,
) -> DataStore:
    """Construct the configured datastore."""

    if str(backend) == "duckdb":
        return DuckDBStore(data_dir=data_dir, db_path=db_path)
    return ParquetStore(
        data_dir=data_dir,
        base_timeframe=base_timeframe,
        resample_source=resample_source,
        derived_dir=derived_dir,
        partial_last_bucket=partial_last_bucket,
        config_hash=config_hash,
        resolved_end_ts=resolved_end_ts,
    )


def _filter_frame(
    frame: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    mask = pd.Series(True, index=data.index)
    if start is not None:
        mask &= timestamps >= _as_timestamp(start)
    if end is not None:
        mask &= timestamps <= _as_timestamp(end)
    return data.loc[mask].reset_index(drop=True)


def _as_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _stringify_ts(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return _as_timestamp(value).isoformat()
