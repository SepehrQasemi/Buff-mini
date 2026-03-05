# Stage-34 ML Dataset Spec

## Scope
- symbols: `['BTC/USDT', 'ETH/USDT']`
- timeframes: `['15m', '30m', '1h', '4h']`
- horizons_hours: `[24, 72]`
- rows_total: `507726`
- max_features: `120` (active=35)
- max_rows_per_symbol: `300000`

## Feature List
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

## Label Definitions
- `label_primary`: primary triple-barrier-like direction label in {-1,0,1}.
- `label_auxiliary`: forward adverse excursion proxy.

## Row Counts
- `BTC/USDT|15m`: `140084`
- `BTC/USDT|1h`: `35008`
- `BTC/USDT|30m`: `70033`
- `BTC/USDT|4h`: `8738`
- `ETH/USDT|15m`: `140084`
- `ETH/USDT|1h`: `35008`
- `ETH/USDT|30m`: `70033`
- `ETH/USDT|4h`: `8738`

## Sampling / Limiting
- Time-consistent truncation only (`tail(max_rows_per_symbol)`), no random sampling.

## No-Leakage Guarantees
- Features are computed from current/past bars only.
- Labels may use forward horizons but are aligned to current timestamp.
- Leakage harness tests include a synthetic intentionally leaky feature check.

## Reproducibility
- config_hash: `55ddf8a401cd`
- data_hash: `b1b6f031a041d620`
- dataset_hash: `ee12b5490af817b4`
- resolved_end_ts: `2026-03-02T00:00:00+00:00`
