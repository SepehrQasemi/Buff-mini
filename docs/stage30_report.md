# Stage-30 Report

## Dataset
- Stage-30.1 implemented deterministic dataset indexing for self-supervised training windows.
- Script: `scripts/build_ml_dataset.py`
- Outputs (run-scoped):
  - `runs/<run_id>/stage30/dataset_index.parquet`
  - `runs/<run_id>/stage30/dataset_meta.json`

## Training
- Stage-30.2 implemented deterministic tiny autoencoder training with CPU fallback.
- Script: `scripts/train_autoencoder.py`
- Core module: `src/buffmini/ml/autoencoder.py`
- Outputs (run-scoped):
  - `runs/<run_id>/stage30/model.pt`
  - `runs/<run_id>/stage30/train_metrics.json`
- Determinism notes:
  - fixed seeds for Python/NumPy/Torch
  - deterministic torch flags enabled when torch backend is used
  - pure NumPy fallback path available (`backend=numpy`) for offline reproducible tests

## Contexts
- Pending Stage-30.3.
