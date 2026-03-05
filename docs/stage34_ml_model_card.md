# Stage-34 ML Model Card

- run_id: `20260305_032548_9d569bce7a2c_stage34_train`
- models: `[{'model_name': 'logreg', 'train_rows': 355408, 'val_rows': 76159, 'test_rows': 76159, 'val_logloss': 0.6920139427765969, 'test_logloss': 0.6937630436615213, 'val_brier': 0.24943319424833374, 'test_brier': 0.2503066472282552, 'prob_mean': 0.49393725329588334, 'prob_std': 0.025584140367411346}, {'model_name': 'hgbt', 'train_rows': 355408, 'val_rows': 76159, 'test_rows': 76159, 'val_logloss': 0.6928780609213754, 'test_logloss': 0.692065555182037, 'val_brier': 0.24986545873178556, 'test_brier': 0.24945922345850857, 'prob_mean': 0.49274353154130085, 'prob_std': 0.001088840100203956}, {'model_name': 'rf', 'train_rows': 355408, 'val_rows': 76159, 'test_rows': 76159, 'val_logloss': 0.6926470824017359, 'test_logloss': 0.6921539628652773, 'val_brier': 0.24974999652409752, 'test_brier': 0.2495034323724239, 'prob_mean': 0.4922132342864906, 'prob_std': 0.0032566111919549376}]`
- calibration: `platt`
- rows_total: `507726`
- split train/val/test: `{'train': 355408, 'val': 76159, 'test': 76159}`

## Hyperparameters
- logreg: gradient descent with L2 regularization
- hgbt: deterministic boosted stump ensemble
- rf: deterministic bagged stump ensemble

## Metrics Summary
- `logreg`: val_logloss=0.692014, test_logloss=0.693763, val_brier=0.249433, test_brier=0.250307
- `hgbt`: val_logloss=0.692878, test_logloss=0.692066, val_brier=0.249865, test_brier=0.249459
- `rf`: val_logloss=0.692647, test_logloss=0.692154, val_brier=0.249750, test_brier=0.249503

## Train/Validation Time Safety
- Time-ordered split only (no shuffle).
- Calibration fitted on validation split only.

## Limitations
- CPU-first lightweight models only.
- Probabilities are calibrated but not guaranteed well-calibrated in sparse contexts.

## Runtime Budget
- Designed for laptop execution with bounded estimators and deterministic seeds.
