# Stage-34 ML Model Card

- rows_total: `507726`
- feature_count: `35`
- split: `{'train': 355408, 'val': 76159, 'test': 76159}`

## Models
- `logreg` val_logloss=0.692014 test_logloss=0.693763 prob_std=0.025584
- `hgbt` val_logloss=0.692878 test_logloss=0.692066 prob_std=0.001089
- `rf` val_logloss=0.692647 test_logloss=0.692154 prob_std=0.003257

## Calibration
- Time-safe calibration on validation split only.

## Runtime
- CPU deterministic training with bounded estimators.
