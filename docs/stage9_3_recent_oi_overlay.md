# Stage-9.3 Recent OI Overlay

- generated_at_utc: `2026-02-28T03:54:43.404865+00:00`
- report_windows_days: `[30, 60, 90]`
- nan_policy: `condition_false`

## OI Availability Facts

### BTC/USDT
- earliest_oi_ts=`2026-01-28T02:00:00+00:00`, latest_oi_ts=`2026-02-26T09:00:00+00:00`, row_count=`704`, raw_rows=`27657`

| requested_days | clamped_days | start_ts | end_ts | note | oi_active_% | oi_active_rows | total_rows |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 30 | 29 | 2026-01-28T02:00:00+00:00 | 2026-02-26T09:00:00+00:00 | CLAMPED | 2.5455 | 704 | 27657 |
| 60 | 29 | 2026-01-28T02:00:00+00:00 | 2026-02-26T09:00:00+00:00 | CLAMPED | 2.5455 | 704 | 27657 |
| 90 | 29 | 2026-01-28T02:00:00+00:00 | 2026-02-26T09:00:00+00:00 | CLAMPED | 2.5455 | 704 | 27657 |

#### A/B Non-Corruption (Non-OI Baseline)
- strategy=`Trend Pullback`, trade_count_delta_pct=`0.000000`, equity_curve_identical=`True`

#### Recent-Window Effects (oi_active only)
| requested_days | condition | horizon | count | median_diff | ci_low | ci_high |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 30 | crowd_long_risk | 24h | 0 | 0.000000 | 0.000000 | 0.000000 |
| 30 | crowd_long_risk | 72h | 0 | 0.000000 | 0.000000 | 0.000000 |
| 30 | crowd_short_risk | 24h | 1 | 0.018331 | 0.016222 | 0.020318 |
| 30 | crowd_short_risk | 72h | 1 | 0.031981 | 0.027470 | 0.035838 |
| 60 | crowd_long_risk | 24h | 0 | 0.000000 | 0.000000 | 0.000000 |
| 60 | crowd_long_risk | 72h | 0 | 0.000000 | 0.000000 | 0.000000 |
| 60 | crowd_short_risk | 24h | 1 | 0.018331 | 0.016362 | 0.020383 |
| 60 | crowd_short_risk | 72h | 1 | 0.031981 | 0.027526 | 0.035874 |
| 90 | crowd_long_risk | 24h | 0 | 0.000000 | 0.000000 | 0.000000 |
| 90 | crowd_long_risk | 72h | 0 | 0.000000 | 0.000000 | 0.000000 |
| 90 | crowd_short_risk | 24h | 1 | 0.018331 | 0.016218 | 0.020319 |
| 90 | crowd_short_risk | 72h | 1 | 0.031981 | 0.027470 | 0.035838 |

### ETH/USDT
- earliest_oi_ts=`2026-01-28T02:00:00+00:00`, latest_oi_ts=`2026-02-26T09:00:00+00:00`, row_count=`704`, raw_rows=`27657`

| requested_days | clamped_days | start_ts | end_ts | note | oi_active_% | oi_active_rows | total_rows |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 30 | 29 | 2026-01-28T02:00:00+00:00 | 2026-02-26T09:00:00+00:00 | CLAMPED | 2.5455 | 704 | 27657 |
| 60 | 29 | 2026-01-28T02:00:00+00:00 | 2026-02-26T09:00:00+00:00 | CLAMPED | 2.5455 | 704 | 27657 |
| 90 | 29 | 2026-01-28T02:00:00+00:00 | 2026-02-26T09:00:00+00:00 | CLAMPED | 2.5455 | 704 | 27657 |

#### A/B Non-Corruption (Non-OI Baseline)
- strategy=`Trend Pullback`, trade_count_delta_pct=`0.000000`, equity_curve_identical=`True`

#### Recent-Window Effects (oi_active only)
| requested_days | condition | horizon | count | median_diff | ci_low | ci_high |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 30 | crowd_long_risk | 24h | 7 | 0.026788 | 0.009890 | 0.031925 |
| 30 | crowd_long_risk | 72h | 7 | 0.026680 | 0.016447 | 0.063862 |
| 30 | crowd_short_risk | 24h | 34 | -0.017785 | -0.033429 | -0.008871 |
| 30 | crowd_short_risk | 72h | 29 | -0.121043 | -0.143276 | 0.050167 |
| 60 | crowd_long_risk | 24h | 7 | 0.026788 | 0.009804 | 0.031925 |
| 60 | crowd_long_risk | 72h | 7 | 0.026680 | 0.016049 | 0.064191 |
| 60 | crowd_short_risk | 24h | 34 | -0.017785 | -0.033368 | -0.009612 |
| 60 | crowd_short_risk | 72h | 29 | -0.121043 | -0.143436 | 0.050182 |
| 90 | crowd_long_risk | 24h | 7 | 0.026788 | 0.009590 | 0.031972 |
| 90 | crowd_long_risk | 72h | 7 | 0.026680 | 0.016418 | 0.064004 |
| 90 | crowd_short_risk | 24h | 34 | -0.017785 | -0.033516 | -0.009490 |
| 90 | crowd_short_risk | 72h | 29 | -0.121043 | -0.143308 | 0.051242 |

## Top Recent-Window Effects
| symbol | requested_days | condition | horizon | count | median_diff | ci_low | ci_high |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| ETH/USDT | 30 | crowd_short_risk | 72h | 29 | -0.121043 | -0.143276 | 0.050167 |
| ETH/USDT | 60 | crowd_short_risk | 72h | 29 | -0.121043 | -0.143436 | 0.050182 |
| ETH/USDT | 90 | crowd_short_risk | 72h | 29 | -0.121043 | -0.143308 | 0.051242 |
| BTC/USDT | 30 | crowd_short_risk | 72h | 1 | 0.031981 | 0.027470 | 0.035838 |
| BTC/USDT | 60 | crowd_short_risk | 72h | 1 | 0.031981 | 0.027526 | 0.035874 |
| BTC/USDT | 90 | crowd_short_risk | 72h | 1 | 0.031981 | 0.027470 | 0.035838 |

## Scope Statement
- OI is unavailable outside the recent overlay window; rows outside oi_active are excluded from OI impact confirmation.

- runtime_seconds: `17.689`
