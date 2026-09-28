# Corrected single-space, matched-artifact run

## Status — verified September 28, 2026

The corrected run is complete and is now the active cross-dataset result bundle.
Oscar GPU array **6368525** completed all **360/360** tasks with exit code `0:0`;
its last task finished September 24 at 03:15:21 EDT. Downstream job **6405567**
completed at 04:55:25 EDT with exit code `0:0` and an empty stderr log. The user
queue was empty at the September 28 check.

All 360 arrays pass strict validation: 121,301 finite scores per prompt, all
required candidates present, and no partial or missing prompts. Local checks
verified every array, metadata, progress, vocabulary, source and frozen-snapshot
hash against the completed run's provenance. The 16,881-row prediction grid
produces 1,986 held-out variant rows covering 993 units in ten datasets.
Recomputed CV, linking tables and Spearman exports match the cluster results.

Top-K selects **K=20 in all ten folds**. Top-p selects **p=0.6 in seven folds**
and **p=0.7 in three**. K no longer reaches the tested ceiling of 100.
The offline viewer and corrected advisor figures now use these results;
its 947 direct neutral candidate scores match the sampling arrays exactly.
All 261 saved viewer metric cells pass numerical parity checks.

See [the completion audit](audit_2026-09-28/REPORT.md),
[aligned tables](linking_structure_tables/LINKING_STRUCTURE_TABLES.md), and
[advisor summary](advisor_summary/ADVISOR_SUMMARY.md). Historical September 8
results remain in `../set_variant_qwen/`, including `results_viewer.html`.

## Locations

All remote paths below are on Oscar.

- Corrected arrays: `/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs_single_space/`
- Historical arrays (unchanged): `/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs/`
- New results: `/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipeline/results/set_variant_qwen_corrected_2026-09-15/`
- Frozen downstream code/input snapshot: `run_snapshot/` inside the new results directory.
- Corrected baseline/linking source: `inputs/source_rows.csv`, generated after all arrays passed validation.
- Full-support target values: `inputs/neutral_candidate_scores.csv`.
- Array and input hashes: `inputs/score_artifact_provenance.json`.
- Framed audit: `inputs/framed_scoring_audit.json` (already passed).

The code/input archive SHA-256 is
`04105a238ba92f9a3d558d3c54572eec62e5e6718d8a596003d75cf196cdb214`.
Its hashes were verified after upload. Local and remote source-row, prompt, and
framed-score inputs match. The archive contains the local code changes for this
run, not a claim that they have been committed. No git commit or push was made.

## What is corrected

Neutral target scores are extracted from the exact float32 candidate arrays
consumed by the sampled linking structures. They are not imported from the old
direct-target scoring CSVs. The exporter records whole-candidate summed scores,
token counts and means, normalized sampling probabilities, and array hashes.
Normalization uses the full 121,301-candidate support, never the target subset.

The direct neutral baseline continues to mean raw continuation probability,
`exp(query_logprob_sum)`. Its relation to full-support normalized probability is
explicitly recorded. Using one score artifact does not conflate these two
probability definitions.

The export rejects incomplete arrays, changed prompts, legacy boundaries,
unexpected model revisions, invalid vocabulary offsets, and nonfinite scores.
The prediction-grid builder rejects mismatched array hashes or target values.
The table builder rejects a direct source file whose hash does not match the
source used by the held-out predictions.

## Framed scorer audit

All 789 X-but-not-Y rows passed, representing 775 unique prompt/candidate pairs.
Each historical prompt already ends in exactly one space. The recorded candidate
token IDs and strings match exact concatenation under the pinned Qwen tokenizer;
the summed and mean scores match the saved token-level scores, and the model
revision and mapping to the analysis rows agree.

These scores are reused unchanged. They have not been relabelled as a new
model-forward run. The audit verifies the scoring convention and provenance,
not an independent numerical rerun of Qwen. The active manifest scorer now
shares the vocabulary scorer's boundary helper and records scorer version 1.2
and boundary version `single-space-exact-concat-v1` for future outputs. Older
checkpoints cannot silently resume under the new version.

## Controls held fixed

This isolates the scoring correction: retain the existing candidate vocabulary,
500 shared weighted orderings per prompt, shortest-prefix original-mass Top-p,
random seed 7, 32,768 retained positions, and grouped ten-fold assignments.
K and p remain separately selected using balanced training-fold Set log score,
then reused for Conjunction/Disjunction in held-out data. No archived
absolute-threshold/Gumbel model is involved.

The historical K and p grids are unchanged in this rerun. Expansion above 100
is an optional sensitivity analysis; the corrected run selects an interior K=20.

## Historical setup verification (September 15)

- 21 targeted scorer/export/validator tests passed.
- 16 grouped-CV and Spearman tests passed.
- Shell syntax and git whitespace checks passed.
- An Oscar pilot on three completed prompt arrays (six candidate scores)
  passed export validation and the direct/sampling hash and value checks.
  The pilot was not used to compute final correlations or fit parameters.
- Full test discovery ran 78 tests with one pre-existing import error:
  `test_set_variant_candidate_vocab.py` imports `build_candidate_vocabs`, which
  does not exist in the unchanged vocabulary-builder module. The other 77
  discovered tests passed. That unrelated mismatch was not modified.

The September 28 sync preserves historical result tables and manuscript drafts.
The frozen snapshot archive is retained as `run_snapshot.tar.gz`; the extracted
`run_snapshot/` remains on Oscar. Large arrays were read for verification and
viewer export and are not added to Git. Changes are local; nothing was pushed.
