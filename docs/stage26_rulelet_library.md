# Stage-26 Rulelet Library

## Scope
- Data source: OHLCV-only.
- No OI/funding dependencies.
- Scores are bounded to `[-1, 1]`.

## Catalog
| Rulelet | Family | Contexts Allowed | Default Exit |
| --- | --- | --- | --- |
| TrendPullback | price | TREND | fixed_atr |
| BreakoutRetest | price | TREND, VOL_EXPANSION | atr_trailing |
| RangeFade | price | RANGE | fixed_atr |
| BollingerSnapBack | price | RANGE | fixed_atr |
| VolCompressionBreakout | volatility | VOL_COMPRESSION | atr_trailing |
| VolExpansionContinuation | volatility | VOL_EXPANSION | atr_trailing |
| MomentumBurst | flow | VOLUME_SHOCK, TREND | fixed_atr |
| MeanRevertAfterSpike | flow | VOLUME_SHOCK, RANGE | fixed_atr |
| StructureBreak | price | TREND | atr_trailing |
| FailedBreakReversal | price | RANGE, VOL_EXPANSION | fixed_atr |
| ChopFilterGate | risk | CHOP | fixed_atr |
| TrendFlip | price | CHOP, TREND | fixed_atr |

## Notes
- `RARE` configurations are allowed in later evaluation but always flagged.
- This catalog is consumed by Stage-26 discovery/replay scripts.
