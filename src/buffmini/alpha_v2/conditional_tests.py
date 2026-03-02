"""Stage-18 conditional edge tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConditionalTestConfig:
    bootstrap_samples: int = 2000
    seed: int = 42
    min_samples: int = 30
    ci_alpha: float = 0.05


def conditional_effects_table(
    *,
    frame: pd.DataFrame,
    signal_col: str,
    context_col: str,
    forward_return_col: str,
    cfg: ConditionalTestConfig,
) -> pd.DataFrame:
    """Compute deterministic conditional effect table."""

    sig = pd.to_numeric(frame.get(signal_col, 0.0), errors="coerce").fillna(0.0)
    ctx = frame.get(context_col, pd.Series("UNKNOWN", index=frame.index)).astype(str)
    fwd = pd.to_numeric(frame.get(forward_return_col, 0.0), errors="coerce").fillna(0.0)
    rng = np.random.default_rng(int(cfg.seed))
    rows: list[dict[str, Any]] = []
    for state in sorted(ctx.unique()):
        mask_state = ctx == state
        on = fwd.loc[mask_state & (sig > 0)]
        off = fwd.loc[mask_state & (sig <= 0)]
        n_on = int(on.shape[0])
        n_off = int(off.shape[0])
        if n_on < cfg.min_samples or n_off < cfg.min_samples:
            rows.append(
                {
                    "context": state,
                    "sample_on": n_on,
                    "sample_off": n_off,
                    "median_diff": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "effect_size": 0.0,
                    "accepted": False,
                    "reason": "INSUFFICIENT_SAMPLE",
                }
            )
            continue
        diff = float(np.median(on) - np.median(off))
        boots = []
        on_arr = on.to_numpy(dtype=float)
        off_arr = off.to_numpy(dtype=float)
        for _ in range(int(cfg.bootstrap_samples)):
            on_s = rng.choice(on_arr, size=on_arr.size, replace=True)
            off_s = rng.choice(off_arr, size=off_arr.size, replace=True)
            boots.append(float(np.median(on_s) - np.median(off_s)))
        ci_low = float(np.quantile(boots, cfg.ci_alpha / 2.0))
        ci_high = float(np.quantile(boots, 1.0 - cfg.ci_alpha / 2.0))
        pooled = np.concatenate([on_arr, off_arr])
        effect_size = float(diff / max(1e-12, np.std(pooled, ddof=0)))
        accepted = bool((ci_low > 0.0) or (ci_high < 0.0))
        rows.append(
            {
                "context": state,
                "sample_on": n_on,
                "sample_off": n_off,
                "median_diff": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "effect_size": effect_size,
                "accepted": accepted,
                "reason": "OK" if accepted else "CI_INCLUDES_ZERO",
            }
        )
    return pd.DataFrame(rows)


def suggest_context_policy(table: pd.DataFrame) -> dict[str, float]:
    """Suggest per-context weight boosts from accepted effects."""

    if table.empty:
        return {}
    out: dict[str, float] = {}
    for row in table.to_dict(orient="records"):
        if not bool(row.get("accepted", False)):
            out[str(row["context"])] = 1.0
            continue
        eff = float(row.get("effect_size", 0.0))
        out[str(row["context"])] = float(np.clip(1.0 + 0.2 * eff, 0.8, 1.2))
    return out


def apply_falsification_rules(
    *,
    table: pd.DataFrame,
    min_samples: int,
) -> pd.DataFrame:
    """Apply deterministic falsification constraints."""

    if table.empty:
        return table.copy()
    out = table.copy()
    out["accepted"] = out["accepted"].astype(bool)
    out["falsified_reason"] = "OK"
    small = (pd.to_numeric(out["sample_on"], errors="coerce").fillna(0.0) < float(min_samples)) | (
        pd.to_numeric(out["sample_off"], errors="coerce").fillna(0.0) < float(min_samples)
    )
    out.loc[small, "accepted"] = False
    out.loc[small, "falsified_reason"] = "SAMPLE_TOO_SMALL"
    ci_includes_zero = (
        pd.to_numeric(out["ci_low"], errors="coerce").fillna(0.0) <= 0.0
    ) & (pd.to_numeric(out["ci_high"], errors="coerce").fillna(0.0) >= 0.0)
    out.loc[ci_includes_zero, "accepted"] = False
    out.loc[ci_includes_zero, "falsified_reason"] = "CI_INCLUDES_ZERO"
    out["multiple_comparisons_warning"] = bool(len(out) > 1)
    return out
