"""Stage-13 signal family contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FamilyContext:
    """Execution context for one family evaluation."""

    symbol: str
    timeframe: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class SignalFamily(ABC):
    """Strict interface for Stage-13 signal families."""

    name: str

    @abstractmethod
    def required_features(self) -> list[str]:
        """Required input columns."""

    @abstractmethod
    def compute_scores(self, df: pd.DataFrame, ctx: FamilyContext) -> pd.Series:
        """Compute signed score in [-1, 1] using only causal inputs."""

    @abstractmethod
    def propose_entries(self, scores: pd.Series, df: pd.DataFrame, ctx: FamilyContext) -> pd.DataFrame:
        """Propose entries with standardized schema."""

    @abstractmethod
    def propose_exits(self, position_state: dict[str, Any], df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        """Suggest exit preferences (must not change engine semantics)."""

    @abstractmethod
    def diagnostics(self, df: pd.DataFrame, ctx: FamilyContext) -> dict[str, Any]:
        """Family diagnostics for reporting."""

    def validate_frame(self, df: pd.DataFrame) -> None:
        required = set(self.required_features())
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{self.name} missing required features: {missing}")

    @staticmethod
    def clip_scores(scores: pd.Series) -> pd.Series:
        arr = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        return pd.Series(np.clip(arr, -1.0, 1.0), index=scores.index, dtype=float)

    @staticmethod
    def build_entry_frame(
        *,
        scores: pd.Series,
        threshold: pd.Series | float,
        family_name: str,
        long_reason: str,
        short_reason: str,
    ) -> pd.DataFrame:
        """Create standardized signal dataframe for downstream engine."""

        clipped = SignalFamily.clip_scores(scores)
        if isinstance(threshold, pd.Series):
            thr = pd.to_numeric(threshold, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            thr = np.clip(thr, 0.0, 1.0)
            thr_series = pd.Series(thr, index=clipped.index, dtype=float)
        else:
            thr_series = pd.Series(float(np.clip(float(threshold), 0.0, 1.0)), index=clipped.index, dtype=float)

        long_entry = (clipped >= thr_series).fillna(False).astype(bool)
        short_entry = (clipped <= -thr_series).fillna(False).astype(bool)
        direction = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        confidence = np.clip(np.abs(clipped.to_numpy(dtype=float)), 0.0, 1.0)
        reasons = np.where(
            direction > 0,
            long_reason,
            np.where(direction < 0, short_reason, ""),
        )
        signal = pd.Series(direction, index=clipped.index, dtype=int).shift(1).fillna(0).astype(int)
        return pd.DataFrame(
            {
                "score": clipped.astype(float),
                "direction": pd.Series(direction, index=clipped.index, dtype=int),
                "confidence": pd.Series(confidence, index=clipped.index, dtype=float),
                "reasons": pd.Series(reasons, index=clipped.index, dtype="object"),
                "long_entry": long_entry,
                "short_entry": short_entry,
                "signal": signal,
                "signal_family": str(family_name),
            }
        )

