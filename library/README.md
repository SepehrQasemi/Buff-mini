# Strategy Library

`library/` stores reusable strategy metadata exported from completed runs.

Contents:

- `index.json`: registry of strategy cards used by the Stage-5 UI.
- `strategies/<strategy_id>/`: compact strategy package (spec/checklist/params/origin).

Rules:

- Keep only lightweight metadata and docs in this folder.
- Do not copy heavy run artifacts or raw market data.
