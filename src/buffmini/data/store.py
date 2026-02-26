"""Backend-ready market data store abstraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.storage import load_parquet, parquet_path, save_parquet


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

    def __init__(self, data_dir: str | Path = RAW_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=self.data_dir)
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
        path = parquet_path(symbol=symbol, timeframe=timeframe, data_dir=self.data_dir)
        if not path.exists():
            return {
                "backend": "parquet",
                "symbol": symbol,
                "timeframe": timeframe,
                "exists": False,
                "rows": 0,
                "start": None,
                "end": None,
                "path": str(path),
            }

        frame = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=self.data_dir)
        if frame.empty:
            return {
                "backend": "parquet",
                "symbol": symbol,
                "timeframe": timeframe,
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
            "timeframe": timeframe,
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
) -> DataStore:
    """Construct the configured datastore."""

    if str(backend) == "duckdb":
        return DuckDBStore(data_dir=data_dir, db_path=db_path)
    return ParquetStore(data_dir=data_dir)


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
