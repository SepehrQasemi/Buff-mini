# Stage-34 ML Dataset Spec

- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['15m', '30m', '1h', '4h']`
- rows_total: `507726`
- data_hash: `b1b6f031a041d620`

## Features
- `ret_1`
- `log_ret_1`
- `ret_3`
- `ret_6`
- `ret_12`
- `atr_14`
- `atr_pct`
- `hl_range_pct`
- `body_ratio`
- `upper_wick_ratio`
- `lower_wick_ratio`
- `vol_12`
- `vol_24`
- `vol_48`
- `volume_z_24`
- `volume_z_72`
- `volume_shock`
- `ma_10`
- `ma_20`
- `ma_50`
- `ma_dist_20`
- `ma_dist_50`
- `ma_slope_20`
- `ma_slope_50`
- `rv_24`
- `rv_72`
- `compression_24`
- `expansion_24`
- `breakout_20`
- `breakdown_20`
- `meanrev_20`
- `tod_sin`
- `tod_cos`
- `dow_sin`
- `dow_cos`

## Labels
- `label_primary`: triple-barrier-like directional label.
- `label_auxiliary`: forward adverse excursion proxy.

## Leakage Safety
- Features are strictly based on current/past bars.
- Labels use future horizon alignment in supervised-learning-safe form only.
