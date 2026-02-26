# Buff-mini Verification Report

## 1) Environment

- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- Key installed packages:
  - `buff-mini` (editable)
  - `ccxt==4.5.36`
  - `numpy==1.26.4`
  - `pandas==2.3.3`
  - `plotly==6.5.2`
  - `pyarrow==14.0.2`
  - `pytest==9.0.2`
  - `PyYAML==6.0.3`
  - `streamlit==1.54.0`

## 2) Test Results

- `python -m compileall src scripts tests`: **PASS**
- `pytest -q`: **PASS** (`16 passed`)
- Stage-0 smoke (`python scripts/run_stage0.py --dry-run`): **PASS**
  - Artifacts created under: `runs/20260226_085004_cdd5d814db48/`
  - Confirmed files include:
    - `config.yaml`
    - `summary.json`
    - `leaderboard.csv`
    - `strategies.json`
    - `summary.csv`
    - strategy-level `trades_*.csv` and `equity_*.csv`

## 3) Coverage Checklist

- Feature leakage tested: **YES**
- Costs tested: **YES**
- Exit priority tested: **YES**
- Reproducibility tested: **YES**
- Artifact integrity tested: **YES**

## 4) Documentation + Configuration Audit

Checked files:
- `README.md`
- `configs/default.yaml`
- `configs/presets/stage0.yaml`
- `configs/presets/quick.yaml`
- `configs/presets/standard.yaml`
- `src/buffmini/config.py`
- `src/buffmini/data/loader.py`
- `scripts/run_discovery.py`

Audit outcomes:
- README command paths verified to exist:
  - `scripts/update_data.py`
  - `scripts/run_stage0.py`
  - `src/buffmini/ui/app.py`
- README implementation claims verified:
  - Binance loader uses `ccxt` in `src/buffmini/data/loader.py`
  - Profitability warning present and clear
- Config keys verified:
  - All 4 YAML configs successfully validated by `load_config` against `validate_config`
- Placeholder clarity verified:
  - `run_discovery` is explicitly documented as a placeholder

Mismatches found and fixed:
- `scripts/run_stage0.py` initially lacked `--dry-run`; added deterministic dry-run mode with synthetic data and full pipeline execution.
- Stage-0 artifacts did not include the expected machine-readable/reporting files; added `summary.json`, `leaderboard.csv`, and `strategies.json`.
- README output section and usage were updated to reflect dry-run support and current artifact set.

## 5) Risk Notes

- Discovery generation remains intentionally unimplemented in MVP Phase 1 (`scripts/run_discovery.py` placeholder).
- Dry-run uses synthetic deterministic market data for pipeline verification; it is not a proxy for real execution quality.
- `funding_pct_per_day` is part of config schema but not yet applied in the current backtest engine.
