# R&X (2024) X-but-not-Y scoring handoff

Completed: GPU job `7864638` scored and validated all 300 rows on October 23,
2026 (exit code 0). CPU integration completed October 26 in
`../results/set_variant_qwen_corrected_rnx_contrastive_2026-10-23/`.
The bundle contains updated tables, paired condition changes, an offline
`results_viewer.html`, and figures under `advisor_summary/four_model_comparison/`.
The original corrected parent and its viewer were preserved; existing neutral
scores, sampled predictions, folds and selected boundaries were reused.

The commands below document reproduction into a new output directory. The
completed score CSV and results bundle are protected from overwrite.

## Inputs and prompt policy

- `rnx_x_but_not_y_manifest.csv`: self-contained scoring input, 300 rows
  (60 items × five human conditions), **180 unique prompt/target pairs**.
- `rnx_x_but_not_y_manifest.json`: source and manifest hashes and coverage.
- `rnx_x_but_not_y_manifest_prompt_audit.csv`: the 180 distinct prompts and
  targets with the conditions and source items sharing each score.

Prompts use the active corrected run's neutral context/QUD prefix, followed by
the actual weaker surface form and ` but not `. Score the actual stronger
surface form. Preserve internal newlines, speaker labels and inflections.
Each prompt ends in exactly one space. Do not append the original sentence
suffix or punctuation after the stronger target. Multi-token targets are scored
in full with summed continuation log probability.

For the good/excellent movie item:

| Conditions | Prompt (newline shown as `\n`) | Target |
| --- | --- | --- |
| ESI, Eonly | `The movie is good but not ` | `excellent` |
| Eweak | `Sue: Is the movie good?\nMary: The movie is good but not ` | `excellent` |
| Estrong, Eonlystrong | `Sue: Is the movie excellent?\nMary: The movie is good but not ` | `excellent` |

**Only is omitted in all five conditions.** ESI/Eonly and
Estrong/Eonlystrong share identical predictions. The QUD can affect predictions.
All 300 human observations remain separate, even though scoring deduplicates
identical prompt/target pairs.

The manifest was derived from
`results/set_variant_qwen_corrected_2026-09-15/inputs/source_rows.csv`.
The cluster scorer needs only this repository, the committed manifest, and the
cached model. The sibling `experiment_ronai&xiang` repository is not required.
Do not rebuild or rescore the existing neutral distributions.

## Cluster preparation and dry run

For transfer, `output/cluster/rnx_x_but_not_y_cluster_ready_2026-10-23.tar.gz`
is an overlay archive containing the manifest, prompt audit, scorer, validator,
wrapper, and updated importer/reporting code. Extract it into an existing
`structure-of-alternatives` checkout; it contains no model weights, human data,
existing results or new model scores. The scoring portion is self-contained;
postprocessing uses the checkout's corrected parent results and viewer.

Run from the `structure-of-alternatives` repository root after transferring the
new manifest, scorer modules and wrapper. Use your existing Qwen environment
(Python 3.10+, NumPy, pandas, torch, transformers, and accelerate for automatic
device mapping). Pin the same Qwen2-7B snapshot as the corrected analysis:

```bash
export QWEN_MODEL_PATH='/path/to/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e'
export PYTHON_BIN=python

bash focus_alt_exp_pipeline/cluster/score_rnx_x_but_not_y_qwen.sh --dry-run
```

The wrapper requires the exact snapshot directory, rather than a mutable cache
reference. Dry run validates the manifest and prints paths/counts; it does not
load model weights or write scores. A real run requires that the snapshot is
already cached; downloads are disabled by default.

Expected counts: 300 rows, five conditions, 180 unique prompts and 180 unique
prompt/candidate pairs. The scorer prints the manifest SHA-256; compare with the
committed JSON sidecar. The scoring boundary is `single-space-exact-concat-v1`:
tokenize the prefix plus one leading space plus the whole stronger alternative.

## Score in your allocated GPU session

```bash
bash focus_alt_exp_pipeline/cluster/score_rnx_x_but_not_y_qwen.sh
```

This runs the existing resumable scorer. **It does not request resources,
invoke sbatch, submit a job, or start postprocessing.** Use your normal cluster
allocation/job workflow to run it when ready.

Defaults:

- Final artifact: `focus_alt_exp_pipeline/model_scores/rnx_x_but_not_y_qwen_scores.csv`
- Checkpoint: `focus_alt_exp_pipeline/model_scores/rnx_x_but_not_y_qwen_scores.partial.csv`
- Precision: bfloat16; device map: auto.
- Query-only scoring; trigger-score fields stay empty.

Rerun the same command to resume an interrupted checkpoint. Keep the manifest
unchanged during a run. A completed final output is protected from accidental
overwrite. Optional environment overrides: `PYTHON_BIN`, `QWEN_DTYPE`,
`QWEN_DEVICE_MAP`, `MANIFEST_PATH`, `OUTPUT_PATH`. Relative paths are resolved
from the repository root by the wrapper; use absolute paths for external files.

The artifact retains token IDs, token strings, token log probabilities, summed
and mean log probability, token count, prompt, target, model revision, runtime
metadata and manifest hash. Return the **final CSV**, not the partial checkpoint.

## After scoring: validate and build the extension (CPU only)

Once the final CSV is present, run from the repository root:

```bash
python focus_alt_exp_pipeline/code/integrate_rnx_x_but_not_y.py \
  --scores focus_alt_exp_pipeline/model_scores/rnx_x_but_not_y_qwen_scores.csv \
  --output-dir focus_alt_exp_pipeline/results/set_variant_qwen_corrected_rnx_contrastive_2026-10-23 \
  --build-reports
```

The importer requires the original corrected results bundle (the default
`results/set_variant_qwen_corrected_2026-09-15/`; override with
`--parent-results` if relocated). It checks all 300 rows, exact manifest and
parent prompt/target alignment, pinned model revision, query-only roles,
corrected token boundary, token counts, token-score sums/means, shared-score
equality and manifest hash. This validates the returned artifact; it does not
independently rerun model inference or retokenize with Qwen.

It refuses to overwrite an existing output or extend a parent that already has
R&X framed scores. It creates a separate bundle containing:

- Extended `inputs/source_rows.csv`, a canonical `all_observations.csv` and a
  300-row `ronai_xiang_2024_observations.csv`.
- Copies of the scoring manifest and returned score artifact.
- Existing prediction grid, held-out predictions, folds and selected boundaries.
  Sampled values are unchanged. Only their source-hash references are updated
  after checking that all neutral inputs/human data were preserved. Original
  serialized neutral source values are retained exactly.
- `inputs/rnx_contrastive_provenance.json`, recording parent/new hashes and the
  precise extension. Original scoring audits are retained with `parent_` names
  and continue to describe the original corrected inputs.
- `rnx_contrastive/paired_condition_changes.csv` and
  `condition_change_summary.csv`: descriptive paired changes for ESI→Eweak,
  Eweak→Estrong, ESI→Eonly and Estrong→Eonlystrong. Both only contrasts must
  have exactly zero model change.

With `--build-reports`, the importer additionally generates aligned Pearson,
Spearman and summed proper-log-score tables, plus a separate
`results_viewer.html` inside the new bundle. It evaluates 60 items separately
within each R&X condition. The prediction is `exp(query_logprob_sum)`, following
the existing X-but-not-Y model, with no pairwise renormalization or calibration.
No sampled-model fitting or boundary selection is repeated.

The new viewer reuses the existing corrected distributions from
`results_viewer/index.html`. It checks the parent/new source and prediction
hashes recorded in the extension audit, exact prompt texts, and original array
hashes before reuse. If relocated, pass `--parent-viewer /path/to/index.html`
to the importer. Alternatively, pass `--log-probs-dir /path/to/corrected/arrays`
to read the existing corrected neutral arrays. This is data export, not model
scoring. New R&X contrastive prompts show their target scores; full framed
vocabulary arrays are not needed for this target-only batch.

Optional four-model comparison figures (requires matplotlib):

```bash
python focus_alt_exp_pipeline/code/plot_four_model_fit_comparison.py \
  --results-dir focus_alt_exp_pipeline/results/set_variant_qwen_corrected_rnx_contrastive_2026-10-23
```

For future canonical rebuilds, `build_cross_dataset_canonical_observations.py`
also accepts `--rnx-x-frame-scores` and `--rnx-x-frame-manifest`. Choose a new
`--output-dir`; that command preserves its historical neutral-source behavior.
Use the importer above to extend the **active corrected** analysis.

## Rebuilding and verification before scoring

```bash
python focus_alt_exp_pipeline/code/build_rnx_x_but_not_y_manifest.py
python -m unittest discover -s focus_alt_exp_pipeline/tests -p 'test_rnx_contrastive.py'
bash -n focus_alt_exp_pipeline/cluster/score_rnx_x_but_not_y_qwen.sh
```

The tests use synthetic token scores only in temporary directories. They do not
load Qwen, create research score artifacts or change existing results.

Preparation verification: all eight R&X extension tests and all 26 viewer
JavaScript tests passed. A full synthetic-score rehearsal built the tables and
viewer, preserved all 360 neutral distributions and checked all 261 saved
metric cells. The Python suite ran 96 entries: 95 passed; one pre-existing
test module cannot import the missing `build_candidate_vocabs` function from
`build_set_variant_candidate_vocab.py`. That unrelated module was left unchanged.
The cluster wrapper passed shell syntax and manifest-only dry-run checks.
The local dry run used a placeholder model path; actual model availability and
Qwen inference remain to be checked in the cluster environment.
