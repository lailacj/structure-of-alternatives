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

Qwen support now exists in code, but it is still an in-progress path while the
full sparse precompute and end-to-end Qwen runs are being finished.

## Directory Layout

- `code/`
  Main scripts for sampling, model scoring, experiment running, and plotting.

- `cloze_data/`
  Human cloze probability data used as one next-word distribution source.

- `human_exp_data/`
  Human responses from the focus-alternative negation experiment.

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
files, but the project-level Qwen results are still being built.

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

This file defines the current alternative-structure models:

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
  Produces per-context summary CSVs from the raw trial-level results.

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
- the context-selected bigrams from `ngrams/qwen_bigram_support/`

The main runner consumes those precomputed files through `--dataset qwen`, using exact context-specific ordering probabilities plus cached sampled estimates for `set`, `conjunction`, and `disjunction`.
That said, Qwen should still be treated as an in-progress model path until the
full 16-context precompute and main result set are complete.

### Qwen implementation handoff (April 6, 2026)

The current Qwen implementation status is:

- `ngrams/qwen_bigram_support/` has been built successfully from the new
  context-balanced builder
- every context has `1500` selected bigrams
- the shared global selected-bigram union is `22789`
- the active precompute output directory is now
  `ngrams/qwen_context_balanced_log_probs/`
- the old dense `20M+` bigram scan is no longer the intended path

Important path note:

- the unigram and frequency bigram source files live under `ngrams/frequency_info/`
- the Qwen precompute defaults and the Oscar wrapper were updated to use those
  files explicitly

Smoke-test status:

- the first `mall` smoke-test failed on a missing `vocab_1gram.txt` path and
  that bug has been fixed
- the rerun created:
  - `ngrams/qwen_context_balanced_log_probs/mall.log_probs.npy`
  - `ngrams/qwen_context_balanced_log_probs/mall.meta.json`
  - `ngrams/qwen_context_balanced_log_probs/mall.progress.json`
- as of April 6, 2026 at about 9:06 PM EDT, `mall.progress.json` reported:
  - `1gram.last_line = 71000`
  - `1gram.done = false`
  - `2gram.last_line = 0`
  - `2gram.done = false`

That means the current smoke-test is no longer failing immediately; it is in
the unigram scoring phase and has not yet reached the selected-bigram phase.

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

For Oscar, there is also an array-job wrapper in:

- `oscar_jobs/precompute_qwen_focus_alt.sh`

Check smoke-test progress with:

```bash
cat /users/ljohnst7/data/ljohnst7/ngrams/qwen_context_balanced_log_probs/mall.progress.json
```

If the `mall` smoke-test completes cleanly, the next command is:

```bash
sbatch oscar_jobs/precompute_qwen_focus_alt.sh
```

Summarize results by context from a model-specific result folder:

```bash
python focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py \
  --results-dir focus_alt_exp_pipeline/results/cloze_probability \
  --models cloze
```

Make the current cloze plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/cloze_probability \
  --title "Cloze Probability: Average Log Likelihood by Structure"
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
- Trial-level model-vs-human negation-probability correlation plots
- Split-half analysis of human negation responses

In progress:

- Qwen runner/sampler support from precomputed context-specific continuation probabilities
- Large-scale sparse Qwen precomputation over all 16 contexts from the context-balanced support set
- End-to-end Qwen result generation and plotting

Not implemented yet in this pipeline:

- Uniform next-word baseline
- Combined across-model comparison plots such as cloze vs frequency vs qwen vs uniform on one figure

## Practical Organization Notes

This directory is the best place to start if you are extending the main project.

If you are extending the model-vs-human negation correlation analysis, the natural inputs are:

- model output CSVs from `results/`, using `negation_probability`
- human trial data from `human_exp_data/sca_dataframe.csv`, aggregated by `(story, trigger, query)`

If you are extending the Qwen path further, the natural integration points are:

- `code/precompute_qwen_vocab_log_probs.py` for generating more context files
- `code/samplers.py` for Qwen-specific sampling and approximation logic
- `code/run_experiment.py` for CLI options and defaults
