from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.stage28.policy_v2 import (
    PolicyV2Config,
    build_policy_v2,
    export_policy_to_library,
    render_policy_spec_md,
)


def _finalists_fixture() -> pd.DataFrame:
    rows = []
    contexts = ["TREND", "RANGE"]
    for ctx_idx, context in enumerate(contexts):
        for idx in range(5):
            rows.append(
                {
                    "candidate_id": f"{context.lower()}_{idx}",
                    "candidate": f"Rulelet{idx}",
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "family": "price",
                    "context": context,
                    "context_occurrences": 90 + idx * 5,
                    "trades_in_context": 35 + idx,
                    "exp_lcb": 0.05 + 0.01 * (4 - idx) + 0.005 * ctx_idx,
                    "expectancy": 0.02 + 0.001 * idx,
                    "cost_sensitivity": 0.02 * idx,
                }
            )
    return pd.DataFrame(rows)


def test_stage28_policy_v2_contract() -> None:
    policy = build_policy_v2(
        _finalists_fixture(),
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="abc123",
        config_hash="cfg999",
        cfg=PolicyV2Config(top_k_per_context=3, conflict_mode="net"),
    )

    assert str(policy.get("version")) == "stage28_policy_v2"
    assert str(policy.get("policy_id", "")).startswith("stage28_")
    contexts = dict(policy.get("contexts", {}))
    assert set(contexts.keys()) == {"TREND", "RANGE"}
    for context in ("TREND", "RANGE"):
        ctx_payload = dict(contexts.get(context, {}))
        assert str(ctx_payload.get("status")) == "OK"
        candidates = list(ctx_payload.get("candidates", []))
        assert len(candidates) == 3
        weights = [float(item.get("weight", 0.0)) for item in candidates]
        assert abs(sum(weights) - 1.0) < 1e-9
        assert all(weight >= 0.0 for weight in weights)
        assert all(str(item.get("candidate_id", "")) for item in candidates)

    markdown = render_policy_spec_md(policy)
    assert "Stage-28 Policy Spec" in markdown
    assert "TREND" in markdown


def test_stage28_policy_v2_export_to_library(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    policy = build_policy_v2(
        _finalists_fixture(),
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="abc123",
        config_hash="cfg999",
    )
    exported = export_policy_to_library(policy, library_dir=library_dir)

    out_path = Path(exported["path"])
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert str(loaded.get("policy_id")) == str(policy.get("policy_id"))

    index = json.loads((library_dir / "index.json").read_text(encoding="utf-8"))
    assert isinstance(index.get("strategies"), list)
    policies = list(index.get("policies", []))
    assert policies
    assert any(str(item.get("policy_id", "")) == str(policy.get("policy_id")) for item in policies)

