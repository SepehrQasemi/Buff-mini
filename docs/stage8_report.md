# Stage-8 Report

Stage-8 adds three anti-self-deception foundations:

1. **Walk-forward v2**: standardized train/holdout/forward triplets with robust aggregation and explicit stability classes.
2. **Cost model v2**: deterministic dynamic execution frictions (spread proxy + volatility-scaled slippage + delay).
3. **Leakage harness**: automated future-shock validation across all registered feature columns.

All results below are from **offline synthetic runs** (no network market data fetch).

## What Changed

- `src/buffmini/validation/walkforward_v2.py`
- `src/buffmini/validation/cost_model_v2.py`
- `src/buffmini/validation/leakage_harness.py`
- `scripts/run_stage8_walkforward.py`
- `scripts/run_stage8_cost_sensitivity.py`
- `src/buffmini/backtest/costs.py`
- `src/buffmini/backtest/engine.py`
- `src/buffmini/data/features.py`
- `configs/default.yaml`
- `src/buffmini/config.py`
- `tests/test_stage8_walkforward_v2.py`
- `tests/test_stage8_cost_model_v2.py`
- `tests/test_stage8_leakage_harness.py`

## How To Run

```bash
python scripts/run_stage8_walkforward.py --seed 42 --rows 10080
python scripts/run_stage8_cost_sensitivity.py
pytest -q tests/test_stage8_walkforward_v2.py tests/test_stage8_cost_model_v2.py tests/test_stage8_leakage_harness.py
```

Expected outputs:

- `runs/<run_id>_stage8_wf/walkforward_v2_report.md`
- `runs/<run_id>_stage8_wf/walkforward_v2_summary.json`
- `docs/stage8_cost_sensitivity.md`
- `docs/stage8_cost_sensitivity.json`

## Key Offline Synthetic Results

Walk-forward v2 (`run_id=20260228_004931_049a74814855_stage8_wf`):

- total windows: `4`
- usable windows: `2`
- classification: `INSUFFICIENT_DATA`
- excluded reasons: `{"min_trades": 2}`

Cost sensitivity (synthetic):

- `simple_delay0` final_equity: `9685.40`
- `v2_low_delay0` final_equity: `9484.03`
- `v2_low_delay1` final_equity: `9579.74`
- `v2_high_delay1` final_equity: `9185.53`

Leakage harness:

- features checked: `20`
- leaks found: `0`

## Interpretation

- Walk-forward v2 enforces multi-window evidence and explicitly marks insufficient evidence instead of overstating stability.
- Cost model v2 surfaces execution fragility through deterministic spread/slippage/delay stress.
- Leakage harness is now a regression guard that fails CI if any registered feature starts using future information.

## Summary Snippet

```json
{
  "walkforward_v2": {
    "usable_windows": 2,
    "classification": "INSUFFICIENT_DATA",
    "excluded_reasons": {"min_trades": 2}
  },
  "cost_sensitivity": {
    "simple_vs_v2_delta": -201.3726,
    "delay_impact": 95.7131
  },
  "leakage_harness": {
    "features_checked": 20,
    "leaks_found": 0
  }
}
```
