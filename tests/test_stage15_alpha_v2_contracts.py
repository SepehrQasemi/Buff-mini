from __future__ import annotations

import numpy as np

from buffmini.alpha_v2.contracts import AlphaRole, ClassicFamilyAdapter
from buffmini.data.features import calculate_features
from buffmini.signals.families.price import PriceStructureFamily
from buffmini.validation.leakage_harness import synthetic_ohlcv


def test_classic_adapter_contract_score_bounds_and_explain() -> None:
    raw = synthetic_ohlcv(rows=800, seed=42)
    frame = calculate_features(raw, config={"data": {"include_futures_extras": False}})
    wrapped = PriceStructureFamily(params={"entry_threshold": 0.3})
    adapter = ClassicFamilyAdapter(
        name="price_entry",
        family="price",
        role=AlphaRole.ENTRY,
        wrapped=wrapped,
        params={"symbol": "BTC/USDT", "timeframe": "1h", "seed": 42, "config": {}},
    )
    scores = adapter.compute_score(frame, seed=42, config={})
    assert scores.shape[0] == frame.shape[0]
    assert np.isfinite(scores.to_numpy(dtype=float)).all()
    assert float(scores.min()) >= -1.0 - 1e-12
    assert float(scores.max()) <= 1.0 + 1e-12
    info = adapter.explain(frame.tail(200))
    assert info["adapter"] == "classic_family"
    assert info["wrapped"] == "price"

