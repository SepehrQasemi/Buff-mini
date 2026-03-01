"""Signal family registry for Stage-13."""

from __future__ import annotations

from typing import Any

from buffmini.signals.families.flow import FlowLiquidityFamily
from buffmini.signals.families.price import PriceStructureFamily
from buffmini.signals.families.volatility import VolatilityCompressionFamily


FAMILY_REGISTRY = {
    "price": PriceStructureFamily,
    "volatility": VolatilityCompressionFamily,
    "flow": FlowLiquidityFamily,
}


def family_names() -> list[str]:
    return sorted(FAMILY_REGISTRY.keys())


def build_families(enabled: list[str], cfg: dict[str, Any]) -> dict[str, object]:
    """Build enabled families with per-family params from config."""

    out: dict[str, object] = {}
    stage13_cfg = ((cfg or {}).get("evaluation", {}) or {}).get("stage13", {})
    for name in enabled:
        key = str(name)
        klass = FAMILY_REGISTRY.get(key)
        if klass is None:
            continue
        params = dict(stage13_cfg.get(key, {})) if isinstance(stage13_cfg, dict) else {}
        out[key] = klass(params=params)
    return out

