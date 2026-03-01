"""Stage-15 alpha-v2 contracts and Classic adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from buffmini.signals.family_base import FamilyContext, SignalFamily


class AlphaRole(str, Enum):
    ENTRY = "Entry"
    CONFIRM = "Confirm"
    RISK_GATE = "RiskGate"
    EXIT_MODIFIER = "ExitModifier"
    SIZING_MODIFIER = "SizingModifier"


@dataclass(frozen=True)
class SignalContract(ABC):
    """Contract for alpha-v2 pluggable components."""

    name: str
    family: str
    role: AlphaRole

    @abstractmethod
    def required_features(self) -> list[str]:
        """Required input feature columns."""

    @abstractmethod
    def compute_score(self, df: pd.DataFrame, *, seed: int, config: dict[str, Any]) -> pd.Series:
        """Return score in [-1, 1] using causal columns only."""

    @abstractmethod
    def explain(self, df_tail: pd.DataFrame) -> dict[str, Any]:
        """Short machine-readable explanation payload."""

    @staticmethod
    def clip_scores(values: pd.Series) -> pd.Series:
        arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.Series(np.clip(arr.to_numpy(dtype=float), -1.0, 1.0), index=values.index, dtype=float)


@dataclass(frozen=True)
class ClassicFamilyAdapter(SignalContract):
    """Adapter that exposes Stage-13 family components through alpha-v2 contract."""

    wrapped: SignalFamily
    params: dict[str, Any]

    def required_features(self) -> list[str]:
        return list(self.wrapped.required_features())

    def compute_score(self, df: pd.DataFrame, *, seed: int, config: dict[str, Any]) -> pd.Series:
        ctx = FamilyContext(
            symbol=str(self.params.get("symbol", "UNKNOWN")),
            timeframe=str(self.params.get("timeframe", "1h")),
            seed=int(seed),
            config=config,
            params=self.params,
        )
        score = self.wrapped.compute_scores(df, ctx)
        return self.clip_scores(pd.Series(score, index=df.index, dtype=float))

    def explain(self, df_tail: pd.DataFrame) -> dict[str, Any]:
        ctx = FamilyContext(
            symbol=str(self.params.get("symbol", "UNKNOWN")),
            timeframe=str(self.params.get("timeframe", "1h")),
            seed=int(self.params.get("seed", 42)),
            config=dict(self.params.get("config", {})),
            params=self.params,
        )
        diag = dict(self.wrapped.diagnostics(df_tail, ctx))
        return {"adapter": "classic_family", "wrapped": self.wrapped.name, "diagnostics": diag}


def validate_contract_output(scores: pd.Series) -> None:
    """Validate contract score range and finiteness."""

    arr = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("alpha-v2 contract returned non-finite score")
    if np.any(arr < -1.000001) or np.any(arr > 1.000001):
        raise ValueError("alpha-v2 contract score must remain in [-1,1]")

