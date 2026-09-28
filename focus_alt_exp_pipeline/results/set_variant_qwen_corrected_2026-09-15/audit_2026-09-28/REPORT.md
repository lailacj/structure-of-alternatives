# Oscar completion audit — September 28, 2026

The corrected run completed successfully, its outputs pass validation, and the
local repo now uses it for the current results viewer and reporting. No failed
job, missing array, changed human input, or inconsistent downstream result was
found in the checks below.

## Cluster execution

| Work | Job | Result | Finished (EDT) |
| --- | --- | --- | --- |
| Corrected Qwen scoring | 6368525, tasks 0–359 | 360 COMPLETED, exit 0:0 | September 24, 03:15:21 |
| Export, sampling, CV and tables | 6405567 | COMPLETED, exit 0:0 | September 24, 04:55:25 |

The queue was empty at the September 28 check. The postprocessing stderr file
is empty. A scan of all 720 GPU stdout/stderr files found no Python traceback,
CUDA out-of-memory, Slurm cancellation/time-limit, or segmentation-fault message.
Accounting is saved in `job_accounting.txt`; original postprocessing logs are
in `../slurm_output/`.

## Integrity and numerical checks

- Strict validation: 360/360 complete prompts, 121,301 finite scores per prompt,
  360 required candidate words present, zero partial/missing arrays or problems.
- All 360 array/metadata/progress SHA-256 sets match the export provenance.
  Model revision and single-space boundary metadata agree for every prompt.
  Vocabulary, original/corrected source and frozen snapshot hashes also match.
- Human observations, inclusion flags, prompts, framed scores, fold assignments,
  group membership and tested K/p grids are unchanged from the historical run.
  Only four original neutral log-score fields and their source-file reference
  change; the export adds the corrected provenance and sampling probabilities.
- The prediction grid has 16,881 rows. Held-out predictions have 1,986 variant
  rows for 993 unique analysis units. No duplicate units occur within a variant
  and boundary. Probabilities are finite and in [0,1], groups stay in one fold,
  and Disjunction = Set + Ordering − Conjunction within numerical tolerance.
- Recomputed CV selections, predictions, score summaries, linking tables and
  Spearman exports reproduce **16 cluster CSVs**, with absolute and relative
  tolerances of 1e-12. The Monte Carlo grid was retained from the cluster;
  sampling and Qwen inference were not rerun during this audit.
- The viewer checks all **261** saved metric cells against rebuilt item values.
  Its 360 neutral distributions include top-50 vocabulary words and experimental
  targets. All **947** direct neutral candidate scores agree with the sampling
  arrays, with zero mismatches or missing exports.
- **55 targeted tests passed:** 19 Python viewer/provenance tests, 18 Python
  CV/Spearman/validator tests, and 18 JavaScript rendering/statistics tests.
  The full unrelated test suite was not rerun; the earlier setup record retains
  its known vocabulary-builder import mismatch.
- The corrected viewer overview and correlation heatmap were visually checked.
  Source/documentation whitespace checks passed. The staged prompt CSV retains
  intentional trailing spaces at model-input boundaries, and the original Slurm
  log retains its formatting; these exact-byte artifacts are excluded from the
  whitespace check to preserve provenance hashes.

Machine-readable checks are in `verification.json`. `cluster_file_manifest.json`
records downloaded files before local documentation updates;
`local_result_manifest.json` records the final local result bundle.

## What changed in the results

| Quantity | Historical September 8 run | Corrected run |
| --- | --- | --- |
| Selected K | 100 in all ten folds | 20 in all ten folds |
| Selected p | 0.6 in all ten folds | 0.6 in seven; 0.7 in three |
| Novel-focus Set Top-K Pearson r | 0.593534 | 0.539816 |
| Novel-focus Ordering Pearson r | 0.461991 | 0.472746 |
| Novel-focus Set Top-K mean log score | −1.568946 | −2.297084 |
| Novel-focus Disjunction Top-K mean log score | −1.527754 | −1.519481 |

The old K-ceiling concern no longer applies to the corrected selection.
Ordering still has the best equal-dataset log score among the proposed
structures (−1.460689), below the fold-safe dataset base rate (about −0.634).
This is a model-fit/calibration limitation, not evidence of incomplete jobs.
The aligned tables contain all datasets and predictors; the complete change
ledger is `historical_vs_corrected_metrics.csv`.

The corrected fridge example has six-word Spearman 1.000 and Set Top-K negation
Spearman 0.859512. The overall mean word-ranking Spearman is 0.698423 across
16 contexts. Mask still has constant Set/Conjunction Top-p predictions, so those
within-context correlations are undefined and their means use 15/16 contexts.
X-but-not-Y remains unavailable for the five R&X conditions; vocabulary-wide
framed arrays were not produced by this run. Those expected coverage limits
are retained explicitly in the viewer.

Across-context association tests were recomputed with the existing permutation,
bootstrap and Holm procedures. Their points and tests are saved here as
`context_association_points.csv` and `context_association_tests.csv`. Direct and
sampling word scores now agree, so the two comparison definitions give identical
results. The existing nine-test adjustment and conservative 18-test sensitivity
adjustment remain available. These exploratory tests condition on the estimates;
they do not propagate participant, norming, sampling or CV-refit uncertainty.

## Updated local files

- The corrected bundle now includes matched source scores, candidate exports,
  prediction grid, CV outputs, linking tables, Spearman ranks, cluster logs and
  the frozen snapshot archive.
- Advisor summaries, aggregate figures and item/context plots were regenerated
  from corrected inputs under `../advisor_summary/`.
- `../../../results_viewer/index.html` now displays the corrected run. The
  viewer's default build selects the corrected bundle and matching `inputs/`.
  A local rebuild preserves its verified embedded vocabulary distributions.
- Root/pipeline/viewer guides and AGENTS.md now identify the completed run.
  The advisor summary generator derives its K-ceiling note from the actual grid.
- Historical results remain under `../../set_variant_qwen/`; its archived
  `results_viewer.html` preserves the previous full viewer and regression fixture.

Pre-existing uncommitted source changes were preserved. Manuscript drafts and
older dated diagnostic reports retain their historical values; use the corrected
bundle when revising them. No commit, push, website publication, remote source
edit or new cluster job was made. Large score arrays remain outside Git.

## Rebuild the viewer

From the repository root, with NumPy and pandas installed:

```bash
python focus_alt_exp_pipeline/code/build_results_viewer.py
```

This uses the corrected default and preserves the verified distributions already
embedded in the HTML. To export fresh arrays on Oscar, add:

```bash
--log-probs-dir /users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs_single_space
```

The pipeline README provides corrected source/result paths for regenerating
linking tables, advisor summaries, item plots and Spearman exports.
