"""Stage-11 multi-timeframe specification primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


_TIMEFRAME_UNITS: dict[str, str] = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def timeframe_to_timedelta(value: str) -> pd.Timedelta:
    """Parse compact timeframe strings like 15m, 1h, 4h, 1d."""

    text = str(value).strip().lower()
    if len(text) < 2:
        raise ValueError(f"Invalid timeframe: {value}")
    unit = text[-1]
    if unit not in _TIMEFRAME_UNITS:
        raise ValueError(f"Unsupported timeframe unit: {value}")
    count = int(text[:-1])
    if count <= 0:
        raise ValueError(f"Timeframe count must be > 0: {value}")
    return pd.Timedelta(**{_TIMEFRAME_UNITS[unit]: count})


def timeframe_ratio(base_timeframe: str, layer_timeframe: str) -> float:
    """Return deterministic duration ratio (layer/base)."""

    base = timeframe_to_timedelta(base_timeframe).total_seconds()
    layer = timeframe_to_timedelta(layer_timeframe).total_seconds()
    if base <= 0 or layer <= 0:
        raise ValueError("timeframe duration must be > 0")
    return float(layer / base)


@dataclass(frozen=True)
class MtfLayerSpec:
    name: str
    timeframe: str
    role: str
    features: tuple[str, ...]
    tolerance_bars: int = 1
    enabled: bool = True

    def tolerance_delta(self, base_timeframe: str) -> pd.Timedelta:
        layer_delta = timeframe_to_timedelta(self.timeframe)
        base_delta = timeframe_to_timedelta(base_timeframe)
        bars = max(1, int(self.tolerance_bars))
        return max(layer_delta, base_delta) * bars


@dataclass(frozen=True)
class MtfSpec:
    base_timeframe: str
    layers: tuple[MtfLayerSpec, ...]
    feature_pack_params: dict[str, Any]
    hooks_enabled: dict[str, bool]


def build_mtf_spec(stage11_cfg: dict[str, Any]) -> MtfSpec:
    """Build normalized MTF specification from config payload."""

    mtf_cfg = dict(stage11_cfg.get("mtf", {}))
    base_timeframe = str(mtf_cfg.get("base_timeframe", "1h"))
    layers_cfg = mtf_cfg.get("layers", [])
    if not isinstance(layers_cfg, list):
        raise ValueError("evaluation.stage11.mtf.layers must be a list")

    layers: list[MtfLayerSpec] = []
    for index, raw in enumerate(layers_cfg):
        if not isinstance(raw, dict):
            raise ValueError(f"evaluation.stage11.mtf.layers[{index}] must be mapping")
        name = str(raw.get("name", f"layer_{index}")).strip()
        timeframe = str(raw.get("timeframe", "")).strip()
        role = str(raw.get("role", "features_only")).strip()
        if role not in {"context", "confirm", "exit", "features_only"}:
            raise ValueError(f"Unsupported stage11 layer role: {role}")
        features_raw = raw.get("features", [])
        if not isinstance(features_raw, list):
            raise ValueError(f"evaluation.stage11.mtf.layers[{index}].features must be list")
        features = tuple(str(item) for item in features_raw)
        tolerance_bars = int(raw.get("tolerance_bars", _default_tolerance_bars(base_timeframe, timeframe)))
        enabled = bool(raw.get("enabled", True))
        # Validate timeframe eagerly.
        timeframe_to_timedelta(base_timeframe)
        timeframe_to_timedelta(timeframe)
        layers.append(
            MtfLayerSpec(
                name=name,
                timeframe=timeframe,
                role=role,
                features=features,
                tolerance_bars=max(1, tolerance_bars),
                enabled=enabled,
            )
        )

    hooks_cfg = dict(stage11_cfg.get("hooks", {}))
    hooks_enabled = {
        "bias": bool(hooks_cfg.get("bias", {}).get("enabled", True)),
        "confirm": bool(hooks_cfg.get("confirm", {}).get("enabled", False)),
        "exit": bool(hooks_cfg.get("exit", {}).get("enabled", False)),
    }
    feature_pack_params = dict(mtf_cfg.get("feature_pack_params", {}))
    return MtfSpec(
        base_timeframe=base_timeframe,
        layers=tuple(layers),
        feature_pack_params=feature_pack_params,
        hooks_enabled=hooks_enabled,
    )


def _default_tolerance_bars(base_timeframe: str, layer_timeframe: str) -> int:
    ratio = timeframe_ratio(base_timeframe, layer_timeframe)
    if ratio <= 1.0:
        return 2
    return int(max(1.0, round(ratio)))

