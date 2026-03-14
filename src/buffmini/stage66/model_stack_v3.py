"""Stage-66 ML stack v3 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage34.train import TrainConfig, predict_model_proba, train_stage34_models
from buffmini.utils.hashing import stable_hash


@dataclass(frozen=True)
class OptionalModelSupport:
    xgboost: bool
    lightgbm: bool
    catboost: bool


def detect_optional_model_support() -> OptionalModelSupport:
    def _has(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception:
            return False

    return OptionalModelSupport(
        xgboost=_has("xgboost"),
        lightgbm=_has("lightgbm"),
        catboost=_has("catboost"),
    )


def train_model_stack_v3(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int,
) -> dict[str, Any]:
    if dataset.empty:
        raise ValueError("dataset is empty")
    work = dataset.copy()
    work["label_primary"] = pd.to_numeric(work.get("tp_before_sl_label", 0.0), errors="coerce").fillna(0.0)
    models, train_summary = train_stage34_models(
        work,
        feature_columns=feature_columns,
        cfg=TrainConfig(seed=int(seed), models=("logreg", "hgbt", "rf"), calibration="platt"),
    )
    base_probs = {
        str(name): predict_model_proba(model, work)
        for name, model in sorted(models.items(), key=lambda kv: str(kv[0]))
    }
    stacked = _stack_probabilities(base_probs)
    y = pd.to_numeric(work["label_primary"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    logloss = _logloss(y, stacked)
    brier = float(np.mean((stacked - y) ** 2))
    support = detect_optional_model_support()
    registry = {
        "version": "model_registry_v5",
        "seed": int(seed),
        "feature_columns": [str(v) for v in feature_columns],
        "base_models": sorted(models.keys()),
        "optional_model_support": asdict(support),
        "stacking": {"method": "mean_ensemble", "calibration": "platt"},
        "metrics": {
            "stack_logloss": float(round(logloss, 8)),
            "stack_brier": float(round(brier, 8)),
            "train_summary_hash": str(train_summary.get("train_summary_hash", "")),
        },
    }
    registry["summary_hash"] = stable_hash(registry, length=16)
    return registry


def _stack_probabilities(base_probs: dict[str, np.ndarray]) -> np.ndarray:
    if not base_probs:
        return np.asarray([], dtype=float)
    names = [str(v) for v in sorted(base_probs.keys())]
    mat = np.vstack([np.asarray(base_probs[name], dtype=float) for name in names])
    mean_prob = np.average(mat, axis=0)
    return np.clip(mean_prob, 1e-6, 1.0 - 1e-6)


def _logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

