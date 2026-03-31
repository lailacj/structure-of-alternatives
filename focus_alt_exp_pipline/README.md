# Focus Alternative Experiment Pipeline

This directory contains the active pipeline for testing whether next-word prediction can explain focus-alternative generation in the human negation task.

At a high level, the pipeline does this:

1. Start from human focus-alternative trials in `human_exp_data/sca_dataframe.csv`.
2. Build a next-word distribution for each context.
3. Sample repeated orderings from that distribution.
4. Define a set by taking the first `set_boundary` items in each sampled ordering.
5. Turn those samples into negation probabilities for different alternative-structure models.
6. Compare model predictions to human responses, currently with log likelihood.

Current active next-word sources in this pipeline:

- human cloze probabilities
- frequency counts from Google Ngram-derived vocab/count tables

Qwen preparation code exists in this directory, but Qwen is not yet fully wired into the end-to-end runner.

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

Qwen is conceptually part of the project, but it is not yet exposed as a dataset option in this CLI.

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

The key helper is `_to_log_likelihood`, which converts a model's negation probability and the observed human response into:

- `log_likelihood`
- `negation_probability`
- `probability_query_observed`

### Sampling next-word distributions

- `code/samplers.py`

This file defines the sampler interface and dataset-specific samplers.

Current active samplers:

- `ClozeSampler`
- `FrequencySampler`

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

- `code/precompute_qwen_vocab_log_probs.py`

This script precomputes Qwen continuation log probabilities over the full ngram vocabulary for each prompt context. It is an important bridge toward the planned Qwen-based focus-alternative model, but it is not yet fully integrated into the main experiment runner.

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
python focus_alt_exp_pipline/code/run_experiment.py \
  --dataset cloze \
  --set-boundaries 2,3,4,5 \
  --num-reps 500
```

Run the frequency baseline:

```bash
python focus_alt_exp_pipline/code/run_experiment.py \
  --dataset frequency \
  --set-boundaries 2,3,4,5 \
  --num-reps 500
```

Summarize results by context from a model-specific result folder:

```bash
python focus_alt_exp_pipline/code/summarize_log_likelihood_by_context.py \
  --results-dir focus_alt_exp_pipline/results/cloze_probability \
  --models cloze
```

Make the current cloze plots:

```bash
python focus_alt_exp_pipline/code/plot_results.py \
  --results-dir focus_alt_exp_pipline/results/cloze_probability \
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
- Frequency-based next-word modeling
- Trial-level model-vs-human negation-probability correlation plots
- Split-half analysis of human negation responses

Partially implemented:

- Qwen continuation scoring over a large vocabulary

Not implemented yet in this pipeline:

- Uniform next-word baseline
- Combined across-model comparison plots such as cloze vs frequency vs qwen vs uniform on one figure

## Practical Organization Notes

This directory is the best place to start if you are extending the main project.

If you are extending the model-vs-human negation correlation analysis, the natural inputs are:

- model output CSVs from `results/`, using `negation_probability`
- human trial data from `human_exp_data/sca_dataframe.csv`, aggregated by `(story, trigger, query)`

If you are adding Qwen to the end-to-end pipeline, the natural integration point is:

- a new sampler in `code/samplers.py`
- a new dataset option in `code/run_experiment.py`
