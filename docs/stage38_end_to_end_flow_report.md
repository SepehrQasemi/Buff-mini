# Stage-38 End-to-End Flow Report

## Run Context
- stage28_run_id: `20260306_152631_2a029423a621_stage28`
- stage28_dir: `C:/dev/Buff-mini/runs/20260306_152631_2a029423a621_stage28/stage28`
- trace_hash: `4e4d501fbca6d698`

## Entrypoints
- Streamlit Run Button:
  launch_app.py -> src/buffmini/ui/pages/20_strategy_lab.py -> src/buffmini/ui/components/run_exec.py::start_pipeline -> scripts/run_pipeline.py
- Direct CLI:
  scripts/run_stage28.py -> scripts/stage37_activation_hunt.py -> scripts/run_stage37.py

## Execution Trace
| order | stage | component | action | input_rows | output_rows | key_details |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | entrypoint | scripts/run_stage28.py | cli_invoked | 0 | 0 | - |
| 2 | config | buffmini.config.load_config | config_and_snapshot_resolved | 0 | 0 | coverage_gate_status=OK |
| 3 | feature_and_context | stage28.context_discovery + stage28.budget_funnel | candidate_matrix_and_funnel_selection | 0 | 6 | - |
| 4 | policy | stage28.policy_v2.build_policy_v2 | policy_composed | 6 | 0 | policy_candidate_count=0, policy_context_count=0 |
| 5 | composer | stage28.policy_v2.compose_policy_signal_v2 | candidate_signals_composed_to_final_signal | 87650 | 0 | candidate_rows_active=0, final_signal_nonzero_rows=0, net_score_nonzero_rows=0 |
| 6 | constraints | run_stage28._apply_live_constraints | live_constraints_and_rejects | 0 | 0 | shadow_reject_rows=0 |
| 7 | evaluation | run_backtest + stage28 metrics | research_live_metrics_scored | 0 | 0 | next_bottleneck=cost_drag_vs_signal, verdict=NO_EDGE |

## Artifact Row Counts
- context_matrix: `0`
- feasibility_envelope: `0`
- finalists_stageC: `6`
- policy_trace: `87650`
- selected_candidates_stageA: `13260`
- selected_candidates_stageB: `6`
- shadow_live_rejects: `0`
- usability_trace: `6`

## Suspicious Branch Checks
- active_candidates_vs_final_signal: `false`
- nonzero_net_required_for_signal: `true`

## Conclusion
- This report is generated from runtime artifacts (`summary.json`, funnel CSVs, policy trace, rejects).
- Any count mismatch between activation hunt and engine should be diagnosed via Stage-38 logic audit table.
