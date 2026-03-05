from __future__ import annotations

from pathlib import Path

from buffmini.stage34.model_registry import RegistryEntry, load_registry, select_elites, top_models, upsert_entry
from buffmini.utils.hashing import stable_hash


def test_stage34_registry_write_read_and_top_selection_deterministic(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    rows_hashes: list[str] = []
    for idx in range(6):
        entry = RegistryEntry(
            model_id=f"m_{idx}",
            generation=idx // 2,
            seed=42,
            symbol="BTC/USDT",
            timeframe="1h",
            horizon="24h",
            feature_subset_sig=f"fset_{idx}",
            hyperparameters={"alpha": 0.1 + idx * 0.01},
            metrics={"exp_lcb": -0.01 + idx * 0.01, "positive_windows_ratio": 0.4 + idx * 0.05, "maxdd_p95": 0.3 - idx * 0.01},
            data_hash="deadbeef",
            resolved_end_ts="2026-03-02T00:00:00+00:00",
            parent_model_ids=(),
        )
        rows = upsert_entry(registry, entry)
        rows_hashes.append(stable_hash(rows, length=12))
    assert len(set(rows_hashes)) == len(rows_hashes)
    loaded = load_registry(registry)
    assert len(loaded) == 6
    top = top_models(registry, top_k=2)
    assert top.shape[0] == 2
    assert top.iloc[0]["model_id"] == "m_5"
    elites = select_elites(registry, generation=2, elite_count=3)
    assert len(elites) == 3
    assert elites[0]["model_id"] == "m_5"


def test_stage34_registry_upsert_deduplicates_model_id(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    first = RegistryEntry(
        model_id="m_same",
        generation=0,
        seed=1,
        symbol="BTC/USDT",
        timeframe="1h",
        horizon="24h",
        feature_subset_sig="a",
        hyperparameters={"x": 1},
        metrics={"exp_lcb": -0.1},
        data_hash="h1",
        resolved_end_ts=None,
    )
    second = RegistryEntry(
        model_id="m_same",
        generation=1,
        seed=1,
        symbol="BTC/USDT",
        timeframe="1h",
        horizon="24h",
        feature_subset_sig="b",
        hyperparameters={"x": 2},
        metrics={"exp_lcb": 0.2},
        data_hash="h1",
        resolved_end_ts=None,
    )
    upsert_entry(registry, first)
    upsert_entry(registry, second)
    loaded = load_registry(registry)
    assert len(loaded) == 1
    assert loaded[0]["generation"] == 1
    assert loaded[0]["metrics"]["exp_lcb"] == 0.2
