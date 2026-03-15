# Buff-mini

Buff-mini is a local, evidence-disciplined crypto research engine for discovering, screening, validating, and diagnosing strategy candidates under reproducible research controls.

It is built to answer two questions at the same time:
- can the system reject fake edges and misleading validation theater?
- can the system still surface weak-but-real signal when controlled signal structure actually exists?

## What Buff-mini Is

- A research system for strategy discovery, ranking, validation, diagnostics, and campaign reporting.
- A local-first Python 3.11 codebase with reproducible artifacts under `docs/` and `runs/`.
- A system that distinguishes exploratory work from interpretation-grade evaluation.
- A repo with PR-based governance and protected `main`.

## What Buff-mini Is Not

- Not a live trading bot.
- Not proof that a profitable live edge already exists.
- Not a system that treats proxy metrics, synthetic placeholders, or pretty reports as decision-grade evidence.
- Not a guarantee that a candidate surviving a heuristic filter should be trusted.

## Current Maturity State

Buff-mini is materially stronger than the original MVP-era repo:
- scientific honesty and stage semantics were tightened in Stages 74-75
- controlled signal detectability, evaluation discipline, mechanism coverage, layered robustness, transfer diagnostics, and research ops were improved in Stages 76-84
- reality isolation, family audit, deeper mechanism registry, funnel diagnostics, data fitness, campaign comparison, scope ladder, transfer intelligence, failure-driven learning, and first interpretable edge inventory were added in Stages 85-94

Current blunt status:
- controlled detectability is proven in synthetic environments
- exploratory discovery and ranking are stronger and more informative than before
- evaluation-mode live campaigns are still blocked by local data continuity issues on the available BTC/ETH datasets
- no robust live edge candidate has been established yet

## Core Principles

### Scientific Honesty
- proxy-only, synthetic, and reporting-only evidence must not drive final promotion
- missing or blocked evidence is surfaced explicitly rather than hidden behind green-looking status
- final verdicts can be blocked, and that is the correct outcome when evidence is insufficient

### Exploration vs Evaluation
- exploration is for search, diagnostics, and hypothesis shaping
- evaluation is for interpretation-grade evidence only
- exploratory outputs are not trustworthy by default for edge claims

### Evidence and Provenance Discipline
Decision-driving artifacts are expected to carry provenance such as:
- candidate id
- run id
- config hash
- data hash
- metric source type
- stage origin
- decision-use flag
- execution status
- validation state

### PR-Based Workflow
- work happens on branches prefixed with `codex/`
- `main` is protected
- pull requests are required before merge
- direct push to `main` is not the workflow

## Architecture Overview

### 1. Discovery and Generation
Key areas:
- `src/buffmini/stage51/`
- `src/buffmini/stage52/`
- `src/buffmini/stage70/`
- `src/buffmini/research/mechanisms.py`

Responsibilities:
- research scope resolution
- candidate schema construction
- mechanism-family expansion
- bounded failure-driven search adaptation
- economic fingerprints and similarity collapse

### 2. Ranking and Funnel
Key areas:
- `src/buffmini/stage48/`
- `src/buffmini/research/behavior.py`
- `src/buffmini/research/diagnostics.py`
- `src/buffmini/research/funnel.py`

Responsibilities:
- leakage-safe tradability labels
- candidate-specific risk cards
- behavioral fingerprints
- candidate hierarchy (`junk`, `interesting_but_fragile`, `promising_but_unproven`, `validated_candidate`, `robust_candidate`)
- funnel-pressure, near-miss, and kill-point diagnostics

### 3. Validation
Key areas:
- `src/buffmini/validation/`
- `src/buffmini/research/robustness.py`
- `src/buffmini/research/transfer.py`
- `scripts/run_stage57.py`, `scripts/run_stage58.py`, `scripts/run_stage67.py`, `scripts/run_stage72.py`

Validation layers include:
- real replay
- walk-forward validation
- Monte Carlo / stress
- cross-perturbation and split perturbation
- transfer validation
- layered robustness classification

### 4. Data and Canonicalization
Key areas:
- `src/buffmini/data/`
- `src/buffmini/research/data_fitness.py`
- `data/snapshots/`
- `data/canonical/`

Responsibilities:
- raw and derived OHLCV loading
- continuity checks and gap reporting
- canonical snapshot metadata
- live vs canonical comparison
- evaluation blocking when strict data prerequisites fail

### 5. Diagnostics and Reporting
Key areas:
- `src/buffmini/diagnostics/`
- `src/buffmini/research/reality.py`
- `src/buffmini/research/family_audit.py`
- `src/buffmini/research/scope_ladder.py`
- `src/buffmini/research/learning.py`
- `docs/`

Responsibilities:
- reality matrix
- family coverage audit
- data fitness reports
- transfer intelligence
- failure taxonomy and search feedback
- campaign inventory reports

### 6. Campaign Execution and Research Operations
Key areas:
- `src/buffmini/research/campaign.py`
- `src/buffmini/research/ops.py`
- `scripts/run_stage83.py`
- `scripts/run_stage94.py`
- `.github/workflows/ci.yml`

Responsibilities:
- bounded multi-scope campaign execution
- run registry and review checklists
- CI semantic smoke checks
- interpretation-grade reporting artifacts

### 7. UI
Key areas:
- `src/buffmini/ui/app.py`
- `src/buffmini/ui/pages/21_run_monitor.py`
- `src/buffmini/ui/pages/22_results_studio.py`

Responsibilities:
- run monitoring
- results browsing
- evidence-quality / runtime-truth display
- reducing false confidence in artifact review

## Stage Roadmap Summary

This repository contains many historical stage scripts. The current repaired story is best understood in these groups:

### Stages 74-75: Semantic Repair
- evidence-source typing
- decision authority cleanup
- transfer gating repair
- honest status semantics
- PR-based workflow hardening

### Stages 76-84: Detectability, Evaluation Discipline, and Research Operations
- synthetic truth lab
- evaluation-mode controls
- mechanism-family redesign
- candidate-specific ranking
- layered robustness funnel
- transfer matrix
- failure-driven feedback
- research ops registry/checklists
- first serious edge campaign path

### Stages 85-94: Diagnosis, Scope, Data Fitness, and Interpretable Inventory
- reality matrix and gate sensitivity
- family coverage audit
- deeper mechanism registry and anti-overlap
- funnel pressure diagnosis
- data fitness and canonical comparison
- evaluation campaign rerun matrix
- scope expansion ladder
- transfer and regime intelligence
- traceable failure-learning loop
- first interpretable edge inventory campaign

## Run Modes

Buff-mini uses explicit run-mode semantics.

### `smoke`
Purpose:
- fast local sanity checks
- CI-friendly basic execution

Interpretable:
- no

### `exploration`
Purpose:
- search
- funnel diagnostics
- mechanism coverage work
- early campaign probing

Interpretable:
- not by default

### `evaluation`
Purpose:
- interpretation-grade comparison attempts
- strict continuity
- frozen/canonical controls
- resolved end pinning

Interpretable:
- only if prerequisites are actually satisfied and the run is not blocked

### `audit`
Purpose:
- post-run review
- semantic or evidence checks
- artifact quality inspection

Interpretable:
- depends on the underlying run mode and evidence state

## Validation Philosophy

### Replay
Replay is the first real gate.
- must use real candles
- must produce real metrics
- can still fail immediately on trade count or expectancy lower-confidence bound

### Walk-Forward
Walk-forward is not decorative.
- forward windows must be real and artifact-backed
- usable-window counts matter
- worst-window and degradation matter more than a single headline metric

### Monte Carlo and Stress
Stress is not a placeholder.
Current stress layers include bounded versions of:
- trade-order shuffle
- cost stress
- timing stress
- split perturbation
- related perturbation variants

### Cross-Perturbation
Cross-perturbation is used as robustness intelligence.
It is not allowed to masquerade as stronger evidence than it really is.

### Transfer
Transfer is a real diagnostic layer.
It can classify candidates as:
- `transferable`
- `partially_transferable`
- `source_local`
- `regime_local`
- `not_transferable`

### What Counts as Decision-Grade Evidence
Decision-grade evidence requires real validation artifacts from the real path.
Proxy-only or synthetic evidence is not sufficient.
A green status on an orchestration or reporting stage is not enough.

## Data Philosophy

### Live vs Canonical
- live data is useful for exploration and operational diagnostics
- canonical snapshot-backed data is preferred for evaluation-grade reruns
- the repo records snapshot metadata and data hashes where possible

### Continuity
Continuity is a first-class research constraint.
- gaps are measured, surfaced, and can block evaluation
- this is desirable because continuity problems can invalidate comparisons

### Why Runs Can Be Blocked
A run can be blocked because of:
- missing resolved end timestamp
- strict continuity failure
- runtime truth failure
- missing or mismatched canonical conditions

That is not a bug by itself. It is part of the honesty contract.

## Current Known Limitations

- No robust live edge candidate has been established yet.
- The current local BTC/ETH evaluation datasets are still blocked by strict continuity gaps under evaluation mode.
- Canonical snapshot metadata exists, but the current canonical data rows do not yet match the snapshot candle counts cleanly.
- Transfer scope is still constrained by locally available assets.
- Generator/search quality is materially better than before, but still bounded and rule-based.
- Ranking is far more candidate-specific than earlier versions, but it is still heuristic and not a proof stage.
- Some older historical stage ecosystems remain in the repo; the repaired late-stage path is the one to trust for current interpretation.

## Quick Start

### Environment
- Python `3.11`
- local repo checkout
- offline verification supported

Install:

```bash
pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

Run the UI:

```bash
streamlit run src/buffmini/ui/app.py
```

## Key Commands

### Baseline / Prerequisite Status
```bash
python scripts/run_baseline_status.py --docs-dir docs
```

### Controlled Detectability Proof
```bash
python scripts/run_stage76.py --config configs/default.yaml --docs-dir docs
```

### Evaluation Mode Check
```bash
python scripts/run_stage77.py --config configs/default.yaml --docs-dir docs --mode evaluation
```

### Family Audit and Refinement
```bash
python scripts/run_stage86.py --config configs/default.yaml --docs-dir docs
python scripts/run_stage87.py --config configs/default.yaml --docs-dir docs
```

### Funnel / Data / Campaign Diagnostics
```bash
python scripts/run_stage88.py --config configs/default.yaml --docs-dir docs
python scripts/run_stage89.py --config configs/default.yaml --docs-dir docs
python scripts/run_stage90.py --config configs/default.yaml --docs-dir docs
```

### Scope / Transfer / Learning / Inventory
```bash
python scripts/run_stage91.py --config configs/default.yaml --docs-dir docs --candidate-limit-per-scope 1
python scripts/run_stage92.py --config configs/default.yaml --docs-dir docs --candidate-limit-per-scope 1
python scripts/run_stage93.py --config configs/default.yaml --docs-dir docs
python scripts/run_stage94.py --config configs/default.yaml --docs-dir docs --max-candidates-per-scope 3
```

### Master Reporting
```bash
python scripts/run_master_execution.py --docs-dir docs
```

## Development Workflow

- create a feature branch with the `codex/` prefix
- make local changes
- run targeted verification
- run `python -m pytest -q`
- push branch
- open PR to `main`
- get approval
- merge through PR

Current governance assumptions:
- `main` is protected
- pull request is required before merge
- at least one approval is required
- force-push to `main` is blocked

## Artifact Guide

Generated artifacts live primarily in:
- `docs/` for stage summaries and reports
- `runs/` for run-specific artifacts and stage outputs

Typical files:
- `docs/stageXX_summary.json`: machine-readable stage summary
- `docs/stageXX_report.md`: human-readable stage report
- `docs/master_execution_summary.json`: top-level execution summary
- `docs/master_execution_report.md`: top-level execution report

How to read them:
- read `validation_state`, not only `status`
- check `stage_role` to understand whether a stage is heuristic, validation, reporting, or orchestration
- confirm whether the run was exploratory or evaluation-mode
- treat blocked outcomes as information, not as missing work

## Honest Project Status

Buff-mini can now make three honest claims:
- it can prove controlled signal detectability in synthetic environments
- it can separate exploratory from evaluation-grade runs much more clearly than before
- it can explain why a live campaign is blocked or weak instead of hiding behind a shallow success label

Buff-mini cannot currently claim:
- that a robust live edge has been found
- that the available BTC/ETH evaluation data is continuity-clean
- that transfer robustness is strong across a broad liquid universe
- that every historical stage in the repo is equally trustworthy

If you are reviewing the project today, the correct reading is:
- the engine is much more scientifically honest
- the discovery, ranking, and diagnostic layers are materially stronger
- controlled detectability is demonstrated
- live interpretation is still constrained by data fitness and the absence of a surviving robust candidate
