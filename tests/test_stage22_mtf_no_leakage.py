from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.alpha_v2.mtf import MtfPolicyConfig, apply_mtf_policy, causal_join_bias


def test_stage22_causal_join_is_backward_only_no_future_leak() -> None:
    base_ts = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC")
    base = pd.DataFrame({"timestamp": base_ts, "entry_score": np.sin(np.arange(48) / 3.0)})
    bias = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=12, freq="4h", tz="UTC"), "bias_score": 0.2})
    joined_a = causal_join_bias(base_df=base, bias_df=bias, bias_col="bias_score")

    # Future shock on later HTF bars must not alter early joined rows.
    bias_shock = bias.copy()
    bias_shock.loc[bias_shock.index >= 8, "bias_score"] = 0.9
    joined_b = causal_join_bias(base_df=base, bias_df=bias_shock, bias_col="bias_score")
    safe_end = 30
    a = pd.to_numeric(joined_a["bias_score"], errors="coerce").iloc[:safe_end].to_numpy(dtype=float)
    b = pd.to_numeric(joined_b["bias_score"], errors="coerce").iloc[:safe_end].to_numpy(dtype=float)
    assert np.allclose(a, b, atol=1e-12)


def test_stage22_policy_outputs_bounded_stats() -> None:
    n = 100
    frame = pd.DataFrame(
        {
            "entry_score": np.linspace(-0.6, 0.6, n),
            "bias_score": np.sin(np.arange(n) / 10.0),
        }
    )
    signal, stats = apply_mtf_policy(
        base_df=frame,
        entry_score_col="entry_score",
        bias_score_col="bias_score",
        cfg=MtfPolicyConfig(conflict_mode="net"),
    )
    assert signal.shape[0] == n
    assert float(stats["conflict_rate_pct"]) >= 0.0
    assert float(stats["conflict_rate_pct"]) <= 100.0
    assert float(stats["bias_alignment_rate_pct"]) >= 0.0
    assert float(stats["bias_alignment_rate_pct"]) <= 100.0

