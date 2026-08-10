# Dataset Card — dashboard_v3 nvBench (frozen)

## Scope

`dashboard_v3` is the final frozen nvBench dataset for structured dashboard-design experiments.
It contains 1,281 Train records, 264 Validation records, and 274 held-out Test records.
The 40-item human-evaluation subset is stored separately and is never trainable.

Analytical source fields originate from nvBench. Phase-3 presentation/design annotations were
automatically validated after a human-reviewed enrichment pilot (`29/30` accepted).

## Files

| File | Role | Trainable? |
| --- | --- | --- |
| `train.jsonl` | SFT corpus | **YES** |
| `val.jsonl` | validation-only corpus | **NO gradient updates** |
| `test.jsonl` | held-out nvBench evaluation | **NO** |
| `human_eval_test_items_40.csv` | separate human-evaluation input | **NO** |
| `schema.json` | `GoldItem` + Phase-3 schema contract | — |
| `manifest.json` | freeze metadata and validation gates | — |
| `hashes.json` | SHA-256 integrity hashes | — |
| `reports/` | audit, validation, leakage, and R1 artifacts | — |

## Field lineage

Source-backed analytical fields remain unchanged from nvBench, including goals, KPIs, columns,
provenance, and the analytical mapping/encoding. Deterministically derived fields include task
inference and related normalized source metadata.

The following six presentation/design fields are LLM-generated and automatically validated:

```text
users
context_summary
layout
styling
interactions
rationales
```

Generation provenance: `deepseek-v4-flash-sovereign`, `reasoning_effort=xhigh`, `temperature=0`.
These fields are not human gold, nvBench gold, or expert gold.

## Reproducibility and integrity

Frozen files are write-once artifacts. Verify SHA-256 values against `hashes.json` before use.
For example:

```powershell
Get-FileHash data/frozen/dashboard_v3/train.jsonl -Algorithm SHA256
Get-FileHash data/frozen/dashboard_v3/val.jsonl -Algorithm SHA256
Get-FileHash data/frozen/dashboard_v3/test.jsonl -Algorithm SHA256
Get-FileHash data/frozen/dashboard_v3/human_eval_test_items_40.csv -Algorithm SHA256
```

The repository loader configuration is `src/config/data/dashboard_v3.yaml`. Experiment configs
E01–E04 reference this frozen dataset and do not reference staging files.

## Validation status

- schema validation: 100% on Train, Validation, and Test
- Train+Validation enrichment completeness: 100%
- immutable/source-backed violations: 0
- duplicate item IDs: 0
- Train/Test leakage: 0
- Validation/Test leakage: 0
- Train/Validation source-group leakage: 0
- Test processed during enrichment: 0
- human-eval items processed during enrichment: 0

## Limitations

This is source-backed structured training data, not independent human gold. Human usefulness and
actionability claims require the separate human-evaluation workflow and ratings.
