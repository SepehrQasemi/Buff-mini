"""Stage-11 hook interfaces and deterministic no-op defaults."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class BiasHook(Protocol):
    def __call__(
        self,
        *,
        timestamp: pd.Timestamp,
        symbol: str,
        signal_family: str,
        signal: int,
        base_row: dict[str, Any],
        activation_multiplier: float,
    ) -> float: ...


class ConfirmHook(Protocol):
    def __call__(
        self,
        *,
        timestamp: pd.Timestamp,
        symbol: str,
        signal_family: str,
        signal: int,
        base_row: dict[str, Any],
    ) -> int: ...


class ExitHook(Protocol):
    def __call__(
        self,
        *,
        timestamp: pd.Timestamp,
        symbol: str,
        signal_family: str,
        exit_mode: str,
        trailing_atr_k: float,
        partial_size: float,
        base_row: dict[str, Any],
    ) -> dict[str, Any]: ...


def noop_bias_hook(
    *,
    timestamp: pd.Timestamp,
    symbol: str,
    signal_family: str,
    signal: int,
    base_row: dict[str, Any],
    activation_multiplier: float,
) -> float:
    _ = (timestamp, symbol, signal_family, signal, base_row, activation_multiplier)
    return 1.0


def passthrough_confirm_hook(
    *,
    timestamp: pd.Timestamp,
    symbol: str,
    signal_family: str,
    signal: int,
    base_row: dict[str, Any],
) -> int:
    _ = (timestamp, symbol, signal_family, base_row)
    return int(signal)


def noop_exit_hook(
    *,
    timestamp: pd.Timestamp,
    symbol: str,
    signal_family: str,
    exit_mode: str,
    trailing_atr_k: float,
    partial_size: float,
    base_row: dict[str, Any],
) -> dict[str, Any]:
    _ = (timestamp, symbol, signal_family, base_row)
    return {
        "exit_mode": str(exit_mode),
        "trailing_atr_k": float(trailing_atr_k),
        "partial_size": float(partial_size),
    }


def build_noop_hooks() -> dict[str, Any]:
    return {
        "bias": noop_bias_hook,
        "confirm": passthrough_confirm_hook,
        "exit": noop_exit_hook,
    }

