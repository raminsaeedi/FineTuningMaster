# Dataset Provenance & Lineage Report

Derived (no record mutation) for 64 items. Dataset version tag: `v2.0-sample`.

## By source_type

- `synthetic`: 24
- `realistic_manual`: 20
- `real_public`: 10
- `real_brief`: 10

## By intended_use

- `independent_benchmark`: 30
- `train`: 18
- `independent_eval`: 10
- `validation`: 3
- `internal_diagnostic`: 3

## By label lineage

- `synthetic_generator:TASK_CHART:v2.0-sample`: 24
- `literature_L1:Saket2019+KimHeer2018`: 22
- `none`: 10
- `manual_expert`: 8

## Safety summary

- synthetic items (internal-diagnostic only): 24
- items safe for INDEPENDENT evaluation: 40 (benchmark_v1 + real_briefs_v1)
- items NOT safe for independent claims (synthetic, generator lineage): 24

> Independent-eval-safe items carry a non-generator label lineage (`literature_L1`, `manual_expert`, or `none`). Synthetic items share the `synthetic_generator:TASK_CHART` lineage and are internal diagnostics only.