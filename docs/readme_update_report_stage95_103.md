# README Update Report (Stage 95-103)

## What Changed

The README was updated to reflect the Stage-95 through Stage-103 work that now exists on this branch:
- live usefulness diagnostics and family freezing
- repaired `canonical_eval` BTC/ETH datasets
- relaxed-to-strict bridge
- mechanism saturation plus deterministic compression
- candidate-quality acceleration and top-K truth review
- multi-scope truth campaign
- null-hypothesis attack against trivial baselines
- bounded rescue attempts
- final edge-existence verdict aggregation

## Why The Previous README Was No Longer Accurate

The previous rewrite still described the repo as if strict evaluation was broadly blocked by BTC/ETH data alone. That was no longer fully true after Stage-96 because:
- `canonical_eval` now provides continuity-clean strict evaluation rows for BTC/ETH 30m/1h/4h
- the repo can now run strict canonical campaigns on those rows
- the stronger blocker shifted from pure data availability to candidate weakness under replay, baseline comparison, and rescue attempts

## Reality Captured By The Updated README

The updated README now states:
- live-strict BTC/ETH remains gap-blocked
- canonical-strict BTC/ETH is available through repaired `canonical_eval`
- the current top candidates are replay-fragile under canonical evaluation
- those candidates do not beat simple momentum / mean-reversion reference baselines in Stage-101
- bounded rescue attempts did not rescue the top candidate in Stage-102
- the current strongest verdict is that generator/search formalism is still insufficient in the present scope

## Sections Updated

- current maturity state
- data and canonicalization architecture
- stage roadmap summary
- data philosophy
- current known limitations
- key commands
- honest project status
