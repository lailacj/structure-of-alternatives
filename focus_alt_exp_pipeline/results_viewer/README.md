# Focus Alternatives results viewer

Open **index.html** in any modern browser. It is a self-contained, offline HTML
file: no server, package installation, CDN, or network connection is needed.

The viewer covers the current Qwen sampled-prefix results in
`focus_alt_exp_pipeline/results/set_variant_qwen`: all ten datasets, nine
linking structures, sixteen novel-focus contexts, both fit measures, item
predictions, and exact prompt/candidate scores. It does not mix in the older
absolute-threshold, human-cloze, or frequency analyses.

## Rebuild locally

From the `structure-of-alternatives` repository root:

```bash
python3 focus_alt_exp_pipeline/code/build_results_viewer.py
```

The default build requires only Python 3.8+ and its standard library. It reads
the source manifest and out-of-fold predictions, rebuilds direct and sampled
item predictions, and verifies all 180 dataset-level correlation/log-score
cells against the saved linking tables before writing the HTML.

Editable source files are `viewer.html.in`, `viewer.css`, and `viewer.js` in this
directory. **index.html is generated**; rebuild after editing these files or
updating result artifacts. The script can run from any working directory.
Existing analysis outputs are read, never modified. Use `--output` to choose a
different HTML destination.

Local rebuilds now preserve distributions embedded in the existing output HTML
when the model, source manifest, out-of-fold predictions, prediction grid, and
prompt text still match. This lets you update plots without the cluster arrays
and without losing the top-50 export. Source-array provenance is retained.
If scientific inputs change, the build stops and asks for fresh arrays rather
than silently attaching old scores. `--log-probs-dir` supplies a fresh export.
`--reuse-distributions-from /path/to/previous/index.html` explicitly recovers
distributions from another export. `--discard-distributions` explicitly permits
a candidate-subset-only rebuild. The build prints neutral-prompt coverage.

The September 14 full export was restored from commit `8e9e907` after the local
rebuild in `3cc2bb4` omitted its vocabulary distributions. This recovery preserves
the original scores; it does not apply the subsequent whitespace-boundary
scoring correction. That correction requires regenerating scores and downstream
results on the cluster, as described in the pipeline README.

`viewer.html` is a compatibility link that opens `index.html`; the unfilled
template now uses the `.html.in` extension so it cannot be mistaken for the
finished viewer. If an embedded file preview does not execute JavaScript, open
`index.html` directly in a browser. The generated file also includes readable
saved tables and a startup message instead of an empty page when scripts do
not run.

## Populate the top-50 vocabulary view on Oscar

Activate the normal analysis environment (NumPy is needed for the `.npy` files),
then run from the repository root:

```bash
python focus_alt_exp_pipeline/code/build_results_viewer.py \
  --log-probs-dir /users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs
```

This reads the original `vocab_manifest.json`, vocabulary files, and each neutral
prompt's `.log_probs.npy`, `.meta.json`, and `.progress.json`. It checks prompt
identity, complete progress, vocabulary counts/uniqueness, array shape, and
finite scores. It embeds the **top 50 candidates plus experimental targets** per
prompt. The viewer displays at most 50 rows at once; candidate search also
finds experimental targets outside the top 50. No Qwen rerun is needed.

Copy the resulting `focus_alt_exp_pipeline/results_viewer/index.html` back to
your computer and open it. The HTML includes its data; the arrays need not be
copied with it.

If vocabulary paths were relocated, provide a directory containing their
original basenames:

```bash
python focus_alt_exp_pipeline/code/build_results_viewer.py \
  --log-probs-dir /path/to/qwen_set_variant_log_probs \
  --vocab-dir /path/to/set_variant_qwen_support
```

`--top-candidates 100` embeds a larger searchable subset while retaining the
50-row display limit. `--top-candidates 0` embeds all candidates: this can make
an extremely large HTML file across 360 × 121,301 candidates. Default top-50
export is recommended. Full arrays are loaded one prompt at a time.

Without `--log-probs-dir`, the viewer shows saved trigger/query scores and
explicitly labels these as an experimental subset, not a top-50 distribution.
The checked-in mask diagnostic also supplies its top vocabulary candidates and
full-support normalized probabilities. It is included only when its recorded
source-manifest hash matches the current manifest. Full-array exports replace
that diagnostic view. The active vocabulary arrays are neutral-prompt arrays;
X-but-not-Y prompts show their saved query scores.

## Views and interpretation

### Optional X-but-not-Y vocabulary scores

See [the cluster audit instructions](CLUSTER_FRAMED_SCORES.md) to locate existing
framed vocabulary scores. Import a complete compatible run with
`--framed-log-probs-dir /path/to/framed_scores`, optionally using
`--framed-vocab-dir /path/to/vocabulary` for relocated vocabulary files. Framed
arrays are matched by metadata prompt text rather than viewer-generated IDs.
The build rejects missing or ambiguous matches, and prints separate neutral
and framed coverage counts. A framed-only import preserves the existing neutral
distributions. Query-only CSVs cannot provide vocabulary-wide top-50 rankings.

### Viewer sections

- **Overview:** aggregate comparisons and a dataset × structure heatmap.
- **Datasets:** both metrics, a model-versus-human scatterplot, and item details.
- **Linking structures:** one structure across all datasets and focus contexts.
- **Focus contexts:** all sixteen contexts × nine structures, then item details.
- **Next words & prompts:** exact neutral/framed prompts, top-50 vocabulary
  distributions when supplied, saved experimental candidates, log probabilities,
  raw continuation probabilities, and full-support sampling probabilities.
- **Methods & sources:** definitions, selected boundaries, coverage, input paths,
  and SHA-256 hashes. Every analysis view has a CSV export.

The default aggregate log score preserves the pipeline's equal-weight mean of
dataset means. Within a dataset, each analysis unit has equal weight. Hu
templates are averaged at the probability level before scale evaluation, and
published-rate datasets do not receive invented participant counts. Item
log scores use natural logs and the existing epsilon of `1e-10`.

The saved pipeline has no single aggregate Pearson correlation. The viewer
labels its additional aggregate correlation as the arithmetic mean of defined
within-dataset correlations. An optional pooled view computes the correlation
over individual analysis units. Neither is a significance test or Fisher-z
meta-analysis.

X-but-not-Y is unavailable for all five R&X conditions. Its available-data
aggregate covers five datasets, so the all-ten best-structure ranking excludes
it. The shared-coverage option compares all nine structures on those same five
datasets. Undefined correlation and structurally unavailable predictions stay
distinct. The base-rate reference comes from the saved fold-safe analysis.

Candidate log scores sum token log probabilities for the complete continuation.
Raw `exp(log p)` and the probability normalized over the entire candidate
support are separate quantities. The experimental subset and the displayed
top 50 are never independently normalized. The full-support and saved target
scores are separate scoring artifacts and are displayed separately. Candidate
strings can overlap, so raw probabilities need not sum to one.

Prompt browsing retains all scored source rows, including Hu rows excluded from
the analysis subset. Only retained analysis units contribute to fit summaries.

## Publish automatically with GitHub Pages

The repository workflow `.github/workflows/deploy-results-viewer.yml` publishes
only the generated `index.html`. It does not rebuild on GitHub, so the full
vocabulary distribution export produced on Oscar remains intact.

One-time setup:

1. In the `lailacj/structure-of-alternatives` repository on GitHub, open
   **Settings → Pages → Build and deployment** and set **Source** to
   **GitHub Actions**. Leave the personal website repository's Pages settings
   unchanged.
2. Commit and push the workflow to `main`.
3. Under **Actions**, open **Publish results viewer** and wait for its deployment
   to finish. If no run starts, select **Run workflow → main → Run workflow**.
4. Open `https://lailacj.github.io/structure-of-alternatives/`.
5. Add a permanent link to that URL in the personal website's Markdown page:

   ```markdown
   [Explore the Focus Alternatives results](/structure-of-alternatives/)
   ```

For future updates, rebuild the viewer with full arrays on Oscar and commit/push
the generated `focus_alt_exp_pipeline/results_viewer/index.html` to `main`.
The workflow publishes it automatically. Changes only to CSVs, Python, CSS, or
JavaScript source will not update the published viewer until the HTML is rebuilt
and pushed. No copy in the personal website repository needs updating.

## Verification commands

```bash
python3 -m unittest discover -s focus_alt_exp_pipeline/tests -p test_results_viewer.py
node --test focus_alt_exp_pipeline/results_viewer/viewer.test.cjs
```

The JavaScript test command needs Node.js 18+; Node is not needed to build or
use the viewer.

Python tests compare all saved item predictions/log scores and metric cells,
check source grain and prompt links, test constant correlations and probability
clipping, and verify safe HTML embedding. Full-array tests also check ranking,
target retention, normalization, and invalid inputs; they skip when NumPy is
unavailable. JavaScript checks exercise summary math and render all views with
the actual data in a lightweight document harness (not browser layout tests).

### Within-context Spearman

Building now requires pandas and NumPy. The dedicated Within-context Spearman
page shows six-word ranking agreement, 30-pair negation agreement, average ranks
for ties, and equal-context means with valid/total counts. Inputs are validated
against `human_exp_data/sca_dataframe.csv`; its hash is included in provenance.
Both measures are focus-only and are not pooled across contexts.

The Spearman page includes side-by-side rank plots and a **Show the fridge example**
button (six-word rho 1.000 versus Set Top-K negation rho 0.784). Change context or
negation model to explore the same distinction elsewhere. Rank plots use separate
0–5 and 0–29 axes; tied pair coordinates are aggregated into circles whose area
reflects their multiplicity. Hover text and expandable tables retain pair identities.
The page explains average ranks, participant aggregation, equal pair weighting,
repeated Set predictions, undefined correlations, and averaging across contexts.

### Hu dataset and R&X condition Spearman

Hu and R&X dataset detail pages now display Spearman alongside Pearson and log
score. Selecting **Spearman correlation** switches to ranks of the same matched
items, with a rank table and model comparison bars. The grouping is within each
Hu dataset or R&X condition, with van Tiel templates averaged before ranking.
Focus retains its separate within-context page. The linking-table builder must
run before rebuilding this viewer; 261 saved metric cells are checked for parity.
