# Stage-8 Cost Model V2 Audit

- backward compatibility (simple legacy parity): `True`
- simple_vs_v2_delta (final_equity): `-863.347809`
- delay_impact (v2_low_d1 - v2_low_d0): `108.758092`
- drag_sensitivity_flag: `HIGH`
- finite_check_passed: `True`

## Monotonicity and Bounds
- spread/slippage increase worsens equity: `True`
- higher volatility increases dynamic slippage: `True`
- delay non-improving in controlled uptrend: `False`
- max_total_bps_per_side cap respected: `True`

## Scenario Table (final_equity)
- `simple_d0`: `9703.157010`
- `v2_high_d0`: `8735.526478`
- `v2_high_d1`: `8839.809201`
- `v2_low_d0`: `9110.427117`
- `v2_low_d1`: `9219.185209`
