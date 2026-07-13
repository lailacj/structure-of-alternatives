# Canonical Data

Canonical observation tables provide a stable interface between dataset-specific
preparation and cross-dataset alternative-structure evaluation. Each row contains
one item-condition human exclusion rate plus neutral-frame Qwen scores for the
uttered trigger and queried alternative.

These files do not contain fitted thresholds or Set, Ordering, Conjunction, or
Disjunction predictions. Those quantities must be learned or calculated later,
inside the evaluation folds.

## Novel focus observations

`novel_focus_observations.csv` contains:

- 480 unique context-trigger-query observations
- 16 context groups
- aggregated counts for all 3,408 binary human judgments
- summed and mean-per-token neutral-frame Qwen scores for each query and trigger
- X-but-not-Y query scores as a separate diagnostic

Rebuild it from the repository root with:

```bash
python focus_alt_exp_pipeline/code/build_focus_canonical_observations.py
```

The pre-existing focus score CSV did not store the resolved Qwen snapshot used
to generate it. The canonical table therefore records:

```text
model_revision=unrecorded_existing_artifact
model_provenance_complete=False
```

This is an explicit provenance limitation, not a validation failure. A future
rescore should pass the exact model revision to the builder and replace the
artifact only after score equivalence is checked.
