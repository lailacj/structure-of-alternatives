# Focus Alternative Experiment Pipeline

This directory contains the active pipeline for testing whether next-word prediction can explain focus-alternative generation in the human negation task.

At a high level, the pipeline does this:

1. Start from human focus-alternative trials in `human_exp_data/sca_dataframe.csv`.
2. Build a next-word distribution for each context, or use a shared global baseline.
3. Sample repeated orderings from that distribution.
4. Define a set by taking the first `set_boundary` items in each sampled ordering.
5. Turn those samples into negation probabilities for different alternative-structure models.
6. Compare model predictions to human responses, currently with log likelihood.

Current stable next-word sources in this pipeline:

- human cloze probabilities
- a global Google Ngram frequency baseline shared across all contexts
- Qwen next-word probabilities from the completed sparse precompute path

## Directory Layout

- `code/`
  Main scripts for sampling, model scoring, experiment running, and plotting.

- `cloze_data/`
  Human cloze probability data used as one next-word distribution source.

- `human_exp_data/`
  Human responses from the focus-alternative negation experiment.

- `canonical_data/`
  Validated item-condition tables that join aggregated human responses to
  neutral-frame Qwen query and trigger scores. These tables are model inputs;
  they do not contain fitted thresholds or structure predictions.

- `scoring_manifests/`
  Self-contained, validated rows for cross-dataset Qwen scoring. The committed
  manifest does not require the local sibling data repository at scoring time.

- `model_scores/`
  Cross-dataset Qwen trigger/query score outputs and their provenance notes.

- `results/`
  Raw trial-level outputs, summaries, diagnostics, and plots.

## Core Inputs

### Human focus-alternative experiment data

Primary file:

- `human_exp_data/sca_dataframe.csv`

Important columns:

- `story`: context identifier
- `trigger`: trigger word
- `query`: queried alternative
- `neg`: human negation response for the trial
- `cleaned_trigger`
- `cleaned_query`

The current runner uses the cleaned columns when available.

### Human cloze data

Primary file:

- `cloze_data/all_cloze_prob_data_preprocessed.csv`

Important columns:

- `context`
- `word`
- `cloze_probability`

## Code Map

### Main entrypoint

- `code/run_experiment.py`

This is the CLI for the current negation-modeling pipeline. It currently supports:

- `--dataset cloze`
- `--dataset frequency`
- `--dataset qwen`

`--dataset qwen` is available for runs backed by completed precomputed context
files, and the current project already has completed Qwen result CSVs and plots.

### Experiment loop

- `code/runner.py`

This file is the stable, dataset-agnostic orchestration layer. It:

- standardizes the human experimental data
- prepares contexts
- asks a sampler for repeated samples
- applies each registered alternative-structure model to each human trial
- writes standardized result CSVs

Result columns are:

- `set_boundary`
- `num_reps`
- `context`
- `trigger`
- `query`
- `neg`
- `log_likelihood`
- `negation_probability`
- `probability_query_observed`

### Alternative-structure models

- `code/models.py`

This file defines the legacy top-K alternative-structure models currently wired
into the experiment runner:

- `ordering`
- `set`
- `conjunction`
- `disjunction`
- `always_negate`
- `never_negate`

The key helper is `probability_to_model_result`, which converts a model's negation probability and the observed human response into:

- `log_likelihood`
- `negation_probability`
- `probability_query_observed`

- `code/absolute_threshold_models.py`

This file defines the new absolute-expectedness-threshold formulation of Set,
Ordering, Conjunction, and Disjunction. It computes the four probabilities
analytically from no-frame query and trigger log probabilities plus one shared
threshold. It does not construct a fixed-size top-K set. The mathematical core
and unit tests are complete. `evaluate_absolute_threshold.py` provides the
leave-one-group-out development evaluator, but this path is not connected to
the legacy sampling runner. Existing runner outputs therefore remain legacy
top-K results until the cross-dataset integration is complete.

- `code/canonical_observations.py`

Defines the shared cross-dataset observation schema and validates item keys,
human counts and rates, neutral-frame score consistency, token-level score
metadata, X-but-not-Y applicability, and model provenance.

- `code/build_focus_canonical_observations.py`

Builds `canonical_data/novel_focus_observations.csv` from the focus human data
and existing Qwen trigger/query score artifact. The current source score CSV did
not record its resolved Qwen snapshot, so the generated rows explicitly carry
`model_revision=unrecorded_existing_artifact` and
`model_provenance_complete=False`.

- `code/evaluate_absolute_threshold.py`

Fits one shared absolute threshold on training groups and produces held-out Set,
Ordering, Conjunction, and Disjunction probabilities. The current development
run uses leave-one-context-out evaluation for the novel focus data, fits the
threshold using the Set model, and reuses that threshold for the hybrid
structures. It also reports item-level Pearson/Spearman correlations and
response-level log scores relative to a training-fold intercept baseline.

The current unit-Gumbel-scale results are integration diagnostics, not final
paper results. They reveal substantial overconfidence in the raw log-probability
scale. The noise scale must be given a cross-validated fitting rule before the
log scores are publication-ready.

### Cross-dataset Qwen scoring status

The repository is **ready for the Hu/Ronai-Xiang no-frame Qwen cluster scoring
run**. The committed `scoring_manifests/hu_rnx_no_frame_manifest.csv` contains:

- 309 Hu context rows (all candidate rows are scored before the exact published
  analysis filter is applied)
- 60 R&X ESI rows
- 60 R&X Eweak rows
- 60 R&X Estrong rows
- 60 R&X Eonly rows using the matched ESI prompt with `only` removed
- 60 R&X Eonlystrong rows using the matched Estrong-QUD prompt with `only`
  removed

`code/score_qwen_scoring_manifest.py` scores weak/trigger and strong/query
continuations, records token-level plus summed and mean log probabilities,
requires an exact model snapshot revision, and checkpoints atomically for
safe `--resume` use. The X-but-not-Y generation frame is intentionally absent
from this manifest; it is a separate Hu/focus-only score family and currently
has no alternative-structure variants.

The one unresolved Hu issue is downstream rather than a scoring blocker: the
exact published Hu inclusion filter still needs to be reconstructed. Every Hu
row therefore carries an explicit `pending_hu_exact_filter_*` status so no
unmarked subset can accidentally become the paper analysis.

The novel focus canonical table is complete from existing scores, apart from
the explicitly recorded historical model-revision provenance gap.

### Sampling next-word distributions

- `code/samplers.py`

This file defines the sampler interface and dataset-specific samplers.

Current active samplers:

- `ClozeSampler`
- `FrequencySampler`
- `QwenSampler`

`FrequencySampler` is intentionally context-free. The intended frequency
baseline is a single global distribution built from a top-K Google Ngram
vocabulary plus all experimental query/trigger tokens force-included so they
remain scoreable.

### Analysis and plotting

- `code/summarize_log_likelihood_by_context.py`
  Produces per-context summary CSVs from the raw trial-level results, plus a
  combined across-model plot with one colored dot per alternative structure and
  a black mean dot per next-word model for both average log likelihood and
  negation-probability correlation.

- `code/plot_results.py`
  Produces:
  - a per-model plot of average log likelihood by alternative structure, with one colored dot per context and a black mean dot
  - one negation-probability correlation plot per alternative structure, comparing model negation probability to human negation probability at the trial-type level

- `code/plot_negation_probability_heatmaps.py`
  Makes heatmap-style visualizations of negation probabilities.

- `code/split_half_neg_correlation.py`
  Computes split-half reliability over human negation responses.

### Qwen preparation

- `../building_vocab_from_ngrams/code/build_qwen_bigram_support.py`

This builder derives required tokens directly from `human_exp_data/sca_dataframe.csv`
and creates a neutral, context-balanced bigram support set for Qwen:

- all human bigrams are force-included for their context
- each context gets the same total bigram budget
- context-specific expansions come from first-word families defined by that
  context's required human tokens
- any remaining slots are filled from a shared background pool ranked by the
  corpus conditional score `count(w1 w2) / count(w1)`

- `code/precompute_qwen_vocab_log_probs.py`

This script now defaults to a context-balanced sparse precompute that is sufficient for the Qwen focus-alternative model:

- all unigrams
- the full global union of selected bigrams from `ngrams/qwen_bigram_support/`

The main runner consumes those precomputed files through `--dataset qwen`, using exact context-specific ordering probabilities plus cached sampled estimates for `set`, `conjunction`, and `disjunction`.

### Qwen Status (April 9, 2026)

The current Qwen implementation status is:

- `ngrams/qwen_bigram_support/` has been built successfully from the new
  context-balanced builder
- every context has `1500` selected bigrams
- the shared global selected-bigram union is `22789`
- the active precompute output directory is now
  `ngrams/qwen_context_balanced_log_probs/`
- the old dense `20M+` bigram scan is no longer the intended path
- Qwen now scores the same shared global vocab for every context:
  - `98502` unigrams
  - `22789` union bigrams
  - `121291` total tokens per context

Important path note:

- the unigram and frequency bigram source files live under `ngrams/frequency_info/`
- the Qwen precompute defaults and the Oscar wrapper were updated to use those
  files explicitly

Completed runtime state:

- all 16 contexts now have completed sparse precompute artifacts under
  `ngrams/qwen_context_balanced_log_probs/`
- the Qwen experiment run has completed and wrote:
  - `results/qwen/ordering_results_qwen.csv`
  - `results/qwen/set_results_qwen.csv`
  - `results/qwen/conjunction_results_qwen.csv`
  - `results/qwen/disjunction_results_qwen.csv`
- the Qwen plotting step has also completed and wrote:
  - `results/qwen/plots/average_log_likelihood_by_context_and_structure__qwen.csv`
  - `results/qwen/plots/mean_log_likelihood_by_structure__qwen.csv`
  - `results/qwen/plots/log_likelihood_by_structure_with_context_dots__qwen.png`
  - `results/qwen/plots/negation_probability_correlation_points__qwen.csv`
  - `results/qwen/plots/negation_probability_correlation__set__qwen.png`
  - `results/qwen/plots/negation_probability_correlation__ordering__qwen.png`
  - `results/qwen/plots/negation_probability_correlation__conjunction__qwen.png`
  - `results/qwen/plots/negation_probability_correlation__disjunction__qwen.png`

## Conceptual Pipeline

For a given context:

1. Get a next-word distribution.
2. Sample a full ordering without replacement from that distribution.
3. Treat the top `set_boundary` items as the sampled set.
4. For each human trial `(context, trigger, query, neg)`:
   - `set`: negate if the query is in the sampled set
   - `ordering`: negate if the query is ranked above the trigger
   - `conjunction`: negate if both are true
   - `disjunction`: negate if either is true
5. Aggregate over repeated samples to estimate a model negation probability.
6. Score the observed human response with log likelihood.

## Outputs

Results are now organized by next-word model under `results/<model>/`.

Examples:

- `results/cloze_probability/`
- `results/frequency/`
- `results/qwen/`

The main raw outputs within one model folder follow this pattern:

- `ordering_results_<model>.csv`
- `set_results_<model>.csv`
- `conjunction_results_<model>.csv`
- `disjunction_results_<model>.csv`
- `missing_trials_<model>.csv`

For example:

- `results/cloze_probability/ordering_results_cloze.csv`
- `results/frequency/set_results_frequency.csv`

Plot and summary outputs for a model live in:

- `results/<model>/plots/`

For example:

- `results/cloze_probability/plots/average_log_likelihood_by_context_and_structure__cloze.csv`
- `results/cloze_probability/plots/mean_log_likelihood_by_structure__cloze.csv`
- `results/cloze_probability/plots/log_likelihood_by_structure_with_context_dots__cloze.png`
- `results/cloze_probability/plots/negation_probability_correlation__set__cloze.png`

## Typical Commands

Run the cloze-based model:

```bash
python focus_alt_exp_pipeline/code/run_experiment.py \
  --dataset cloze \
  --set-boundaries 2,3,4,5 \
  --num-reps 500
```

Run the frequency baseline:

```bash
python focus_alt_exp_pipeline/code/run_experiment.py \
  --dataset frequency \
  --num-reps 500
```

By default, `--dataset frequency` now uses
`--frequency-background-vocab-size 800000` unless you override it or use
`--frequency-max-vocab-size`. The default set-boundary sweep is `3, 6, ..., 99`
because the step size is `3`.

Run the Qwen model from precomputed context files:

```bash
python focus_alt_exp_pipeline/code/run_experiment.py \
  --dataset qwen \
  --qwen-top-vocab-size 100000 \
  --num-reps 500
```

For Qwen, each context now samples `num_reps` top-prefix orderings once at the
maximum requested set boundary and reuses those same samples across all smaller
set boundaries and across `set`, `conjunction`, and `disjunction`. `ordering`
is computed exactly from the context-specific Qwen probabilities, and
`disjunction` is computed from `set + ordering - conjunction`.

Build the context-balanced Qwen bigram support files:

```bash
python building_vocab_from_ngrams/code/build_qwen_bigram_support.py \
  --context-bigram-budget 1500 \
  --shared-background-pool-size 500
```

Build the sparse Qwen precompute for one context:

```bash
python focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py \
  --contexts mall \
  --target-vocab-size 100000 \
  --vocab-1gram /users/ljohnst7/data/ljohnst7/ngrams/frequency_info/vocab_1gram.txt \
  --local-files-only \
  --hf-offline
```

Even though the support builder uses context-balanced selection to define the
global union, the Qwen scorer now evaluates that same union vocabulary for every
context prompt rather than scoring a smaller context-specific subset.

For Oscar, there is also an array-job wrapper in:

- `oscar_jobs/precompute_qwen_focus_alt.sh`

Run the full Qwen experiment on Oscar with:

```bash
sbatch oscar_jobs/focus_alt_exp.sh
```

Make the current cloze plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/cloze_probability \
  --model cloze
```

Summarize results by context across the current model result folders:

```bash
python focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py \
  --results-dir focus_alt_exp_pipeline/results \
  --models cloze,frequency,qwen \
  --model-order cloze,frequency,qwen
```

This summary command now also writes:

- `results/plots/mean_log_likelihood_by_model_and_structure.csv`
- `results/plots/mean_log_likelihood_by_model.csv`
- `results/plots/log_likelihood_by_model_with_structure_dots.png`
- `results/plots/negation_probability_correlation_points.csv`
- `results/plots/negation_probability_correlation_by_model_and_structure.csv`
- `results/plots/negation_probability_correlation_by_model.csv`
- `results/plots/negation_probability_correlation_by_model_with_structure_dots.png`

Make the current Qwen plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/qwen \
  --model qwen
```

This plotting command currently writes:

- the per-context average-log-likelihood-by-structure plot
- a trial-level CSV aligning model and human negation probabilities
- one negation-probability correlation scatter plot for each of:
  - `set`
  - `ordering`
  - `conjunction`
  - `disjunction`

Human negation probability is computed as the proportion of participants with `neg = 1` for each `(context, trigger, query)` trial type.

## What Is Implemented vs Planned

Implemented now:

- Trial-level log-likelihood evaluation
- Cloze-based next-word modeling
- Frequency-based next-word modeling with a global Google Ngram baseline
- Qwen next-word modeling from sparse precomputed continuation probabilities
- Trial-level model-vs-human negation-probability correlation plots
- Split-half analysis of human negation responses

Not implemented yet in this pipeline:

- Uniform next-word baseline

## Practical Organization Notes

This directory is the best place to start if you are extending the main project.

If you are extending the model-vs-human negation correlation analysis, the natural inputs are:

- model output CSVs from `results/`, using `negation_probability`
- human trial data from `human_exp_data/sca_dataframe.csv`, aggregated by `(story, trigger, query)`

If you are extending the Qwen path further, the natural integration points are:

- `code/precompute_qwen_vocab_log_probs.py` for generating more context files
- `code/samplers.py` for Qwen-specific sampling and approximation logic
- `code/run_experiment.py` for CLI options and defaults
- `code/plot_results.py` for model-specific result visualization
