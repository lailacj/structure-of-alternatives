# Structure of Alternatives

Research code for modeling focus-alternative generation as a next-word prediction problem.

The central claim of this project is:
if a model is good at predicting what word people expect next in a context, then it should also be good at predicting which alternatives people treat as relevant in a focus-alternative negation task.

This repo currently supports four conceptual next-word sources:

1. A uniform distribution over words. Planned, not yet implemented in the active pipeline.
2. Human cloze probabilities. Implemented.
3. Word-frequency distributions from Google Ngram counts. Implemented.
4. Qwen next-word probabilities conditioned on context. Implemented through the sparse precompute path, with completed result CSVs and plots under `focus_alt_exp_pipeline/results/qwen/`.

## Main Question

For each context shown in the human focus-alternative experiment:

1. Build a next-word distribution.
2. Sample sets and orderings from that distribution.
3. Convert those samples into negation probabilities for several alternative-structure models.
4. Compare model predictions against human responses.

The repository now contains two related evaluation paths:

- the original within-dataset runner for cloze, frequency, and Qwen sources;
- the active cross-dataset sampled-prefix analysis over the novel-focus data,
  four Hu benchmark datasets, and five Ronai-Xiang conditions.

The active cross-dataset analysis uses grouped 10-fold selection and evaluates
item-level proper log score, Pearson correlation, and Spearman correlation.
Spearman uses within-context groups for focus alternatives, within-dataset scales
for the Hu benchmarks, and within-condition items for Ronai–Xiang (2024). The original runner uses
trial-level log likelihood and also includes correlation plots between:

- the model's negation probability for a given `(context, trigger, query)` trial, and
- the human negation probability for that same trial.

## Active Data Sources

The original runner uses two human datasets:

1. Cloze probability data:
   `focus_alt_exp_pipeline/cloze_data/all_cloze_prob_data_preprocessed.csv`

2. Focus-alternative human experiment data:
   `focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv`

Useful columns in the focus-alternative experiment data include:

- `story`: context name
- `trigger`: trigger item in the trial
- `query`: queried alternative
- `neg`: whether participants negated the query on that trial
- `cleaned_trigger`, `cleaned_query`: normalized forms used by the current pipeline

The active cross-dataset analysis reads the validated tables under
`focus_alt_exp_pipeline/canonical_data/`. These combine 480 novel-focus rows,
309 Hu benchmark rows, and 300 Ronai-Xiang rows.

## Where The Main Pipeline Lives

The active negation-modeling pipeline is in:

- `focus_alt_exp_pipeline/`

Important files:

- `focus_alt_exp_pipeline/code/models.py`
  Defines how sampled orderings/sets are turned into negation probabilities for the `ordering`, `set`, `conjunction`, and `disjunction` models. Also contains `_to_log_likelihood`.

- `focus_alt_exp_pipeline/code/samplers.py`
  Builds sampled next-word orderings/sets from different next-word distributions.

- `focus_alt_exp_pipeline/code/runner.py`
  Dataset-agnostic experiment loop. It applies the models to each trial and writes standardized result CSVs.

- `focus_alt_exp_pipeline/code/run_experiment.py`
  Main CLI entrypoint for running the active pipeline.

- `focus_alt_exp_pipeline/code/plot_results.py`
  Summarizes result CSVs and produces log-likelihood plots and diagnostics.

- `focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py`
  Produces per-context summaries from the raw result CSVs.

- `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py`
  Precomputes sparse Qwen continuation log probabilities for each prompt context.
- `building_vocab_from_ngrams/code/build_qwen_bigram_support.py`
  Builds the context-balanced bigram support set that the Qwen precompute consumes.

- `focus_alt_exp_pipeline/code/split_half_neg_correlation.py`
  Estimates split-half reliability for human negation responses.

- `focus_alt_exp_pipeline/code/build_set_variant_prediction_grid.py`
  Samples shared weighted orderings and constructs the Top-K/Top-p grid.

- `focus_alt_exp_pipeline/code/evaluate_set_variant_grid.py`
  Selects K and p separately in grouped training folds and evaluates held-out
  Set, Conjunction, and Disjunction predictions.

- `focus_alt_exp_pipeline/code/build_linking_structure_tables.py`
  Aligns direct Qwen baselines and sampled-prefix models in final cross-dataset
  tables.

There is also a more focused guide at:

- `focus_alt_exp_pipeline/README.md`

## Repo Map

These directories are the ones that matter most right now:

- `focus_alt_exp_pipeline/`
  Main focus-alternative negation-modeling pipeline.

- `next_word_prediction_correlations/`
  Earlier and parallel work comparing next-word predictions from LLMs to human cloze-like distributions. This is useful background for the motivation that Qwen should be a good focus-alternative model.

- `prompts/`
  Prompt templates and generated prompt CSVs used for language-model continuation scoring.

- `building_vocab_from_ngrams/`
  Scripts and data for constructing vocabulary resources from ngram corpora.

- `archive/`
  Older code, results, figures, and writeups that are kept for reference but are not the main working pipeline.

- `hsp_poster_code/`
  One-off poster-era scripts related to Qwen top-word exploration.

## Original Within-Dataset Pipeline Flow

The active focus-alternative workflow is:

1. Load the human focus-alternative trials from `focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv`.
2. Choose a next-word source such as cloze, frequency, or Qwen.
3. Use a sampler to draw repeated orderings of candidate next words.
4. Define the set as the first `set_boundary` items in each sampled ordering.
5. Score each human trial with one or more alternative-structure models from `models.py`.
6. Save trial-level outputs including:
   - `log_likelihood`
   - `negation_probability`
   - `probability_query_observed`
7. Summarize and plot the resulting CSVs.

The result naming convention is:

- `<alternative_structure>_results_<next_word_model>.csv`
- `missing_trials_<next_word_model>.csv`

For example:

- `ordering_results_cloze.csv`
- `set_results_frequency.csv`
- `disjunction_results_qwen.csv`

## Active Cross-Dataset Sampled-Prefix Flow

For each of 360 distinct prompts, Qwen scores a shared candidate support of
98,509 unigrams and 22,792 bounded bigrams. The analysis then:

1. samples 500 weighted orderings without replacement;
2. defines Top-K as the first K words and Top-p as the shortest prefix whose
   original normalized candidate probabilities reach p;
3. selects K and p separately in grouped 10-fold training data using balanced
   Set log score;
4. reuses the selected boundary for Conjunction and Disjunction in the held-out
   fold;
5. reports proper log scores and correlations across 993 analysis units in 10
   dataset strata.

The full-coverage current run is under
`focus_alt_exp_pipeline/results/set_variant_qwen/`.

## Quick Start

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Run the active pipeline with human cloze distributions:

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
  --set-boundaries 2,3,4,5 \
  --num-reps 500
```

Summarize log likelihood by context:

```bash
python focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py \
  --results-dir focus_alt_exp_pipeline/results \
  --models cloze,frequency,qwen \
  --model-order cloze,frequency,qwen
```

Make cloze plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/cloze_probability \
  --model cloze
```

Make Qwen plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/qwen \
  --model qwen
```

Rebuild the cross-dataset prediction grid and grouped-CV results from completed
cluster score arrays:

```bash
sbatch focus_alt_exp_pipeline/cluster/run_set_variant_postprocessing.sh
```

The current run used 500 repetitions, seed 7, and candidate grids
`K = 1,2,3,5,10,20,30,50,75,100` and
`p = 0.25,0.5,0.6,0.7,0.8,0.9,0.95`.

## Current Status

Implemented in the active pipeline:

- Human cloze-based next-word distributions
- Google Ngram frequency-based next-word distributions
- Qwen next-word distributions from sparse precomputed continuation log probabilities
- Trial-level negation probabilities for `ordering`, `set`, `conjunction`, and `disjunction`
- Trial-level log-likelihood evaluation
- Trial-level model-vs-human negation-probability correlation plots
- Plotting and per-context log-likelihood summaries
- Cross-dataset canonical observation tables
- Bounded-support Qwen distribution scoring for all 360 prompts
- Sampled-prefix Top-K and Top-p prediction grids
- Grouped 10-fold boundary selection and out-of-fold evaluation
- Cross-dataset linking-structure tables, heatmaps, and item-level plots

### Original within-dataset Qwen status (April 9, 2026)

The active Qwen path is now complete end-to-end for the focus-alternative
pipeline.

Completed artifacts:

- `ngrams/qwen_bigram_support/` contains the context-balanced bigram support set
- `ngrams/qwen_context_balanced_log_probs/` contains completed sparse precompute outputs for all 16 contexts
- `focus_alt_exp_pipeline/results/qwen/` contains the raw Qwen result CSVs:
  - `ordering_results_qwen.csv`
  - `set_results_qwen.csv`
  - `conjunction_results_qwen.csv`
  - `disjunction_results_qwen.csv`
- `focus_alt_exp_pipeline/results/qwen/plots/` contains the generated Qwen summaries and plots:
  - `average_log_likelihood_by_context_and_structure__qwen.csv`
  - `mean_log_likelihood_by_structure__qwen.csv`
  - `log_likelihood_by_structure_with_context_dots__qwen.png`
  - `negation_probability_correlation_points__qwen.csv`
  - one correlation plot per alternative structure

Key Qwen implementation details that remain true:

- the support builder is `building_vocab_from_ngrams/code/build_qwen_bigram_support.py`
- the sparse precompute is `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py`
- the runner is `focus_alt_exp_pipeline/code/run_experiment.py --dataset qwen`
- the current Oscar wrapper for the Qwen experiment run is `oscar_jobs/focus_alt_exp.sh`
- the Qwen plotting command is `focus_alt_exp_pipeline/code/plot_results.py --results-dir focus_alt_exp_pipeline/results/qwen --model qwen`

### Cross-dataset sampled-prefix status (September 8, 2026)

The full planned scoring coverage and current postprocessing run are complete:

- 360/360 prompt score arrays passed strict validation;
- each prompt contains 121,301 finite candidate scores;
- all 360 required trigger/query candidates are present;
- the prediction grid contains 16,881 rows;
- the out-of-fold results cover 993 analysis units across 10 dataset strata;
- Top-p selected `p=0.6` in all 10 folds;
- Top-K selected `K=100` in all 10 folds.

Because 100 is the largest tested K, the full-coverage run is complete but the
Top-K search is not yet frozen. Expand the K grid before treating K=100 as an
interior optimum or calling the analysis final.

Planned next:

- Uniform next-word baseline
- Expand the Top-K grid beyond 100 and rerun CPU-only postprocessing

## Notes For Future Contributors And AI Agents

- If you are trying to understand the current project, start in `focus_alt_exp_pipeline/code/run_experiment.py`, `focus_alt_exp_pipeline/code/runner.py`, and `focus_alt_exp_pipeline/code/models.py`.
- The active evaluation target is the human negation task, not just next-word agreement.
- The repo still contains older BERT-era work and exploratory scripts; do not assume all top-level directories reflect the current main workflow.
- The active directory is `focus_alt_exp_pipeline`.
- Treat `results/set_variant_qwen/` as the active cross-dataset result bundle.
- Treat `results/qwen/` as the completed original within-dataset Qwen result bundle.
- Do not use `archive/absolute-threshold-gumbel-august-2026/` for the active
  sampled-prefix analysis.
