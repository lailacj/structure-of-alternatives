# Structure of Alternatives

Research code for modeling focus-alternative generation as a next-word prediction problem.

The central claim of this project is:
if a model is good at predicting what word people expect next in a context, then it should also be good at predicting which alternatives people treat as relevant in a focus-alternative negation task.

This repo currently supports four conceptual next-word sources:

1. A uniform distribution over words. Planned, not yet implemented in the active pipeline.
2. Human cloze probabilities. Implemented.
3. Word-frequency distributions from Google Ngram counts. Implemented.
4. Qwen next-word probabilities conditioned on context. In progress: sparse precompute and runner support exist, but the full 16-context precompute and end-to-end Qwen results are still being completed.

## Main Question

For each context shown in the human focus-alternative experiment:

1. Build a next-word distribution.
2. Sample sets and orderings from that distribution.
3. Convert those samples into negation probabilities for several alternative-structure models.
4. Compare model predictions against human responses.

The main implemented evaluation is log likelihood.
The next planned evaluation is correlation between:

- the model's negation probability for a given `(context, trigger, query)` trial, and
- the human negation probability for that same trial.

## Active Data Sources

There are two main human datasets in the active pipeline:

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

## Current Pipeline Flow

The active focus-alternative workflow is:

1. Load the human focus-alternative trials from `focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv`.
2. Choose a next-word source such as cloze or frequency.
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
  --models cloze,frequency
```

Make summary plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --models cloze,frequency
```

## Current Status

Implemented in the active pipeline:

- Human cloze-based next-word distributions
- Google Ngram frequency-based next-word distributions
- Trial-level negation probabilities for `ordering`, `set`, `conjunction`, and `disjunction`
- Trial-level log-likelihood evaluation
- Plotting and per-context log-likelihood summaries

In progress:

- Qwen sparse continuation precompute for the focus-alternative prompts
- Qwen end-to-end runs through the main negation pipeline while the full context precompute is being completed

### Qwen Handoff Status (April 6, 2026)

The Qwen path was recently refactored away from the old global bigram-threshold
builder and toward a neutral, context-balanced sparse support set.

What is finished:

- `building_vocab_from_ngrams/code/build_qwen_bigram_support.py` now builds the
  Qwen bigram support directly from
  `focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv`
- the old threshold-based 2-gram builders were moved into `archive/code_archive/`
- `ngrams/qwen_bigram_support/` has already been built successfully
- each of the 16 contexts has exactly `1500` selected bigrams
- the selected bigram union size is `22789`
- `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py` now scores:
  - all unigrams from `ngrams/frequency_info/vocab_1gram.txt`
  - only the selected context-balanced bigrams from `ngrams/qwen_bigram_support/`

What was fixed during smoke-testing:

- an initial `mall` smoke-test failed because the Qwen precompute script looked
  for `ngrams/vocab_1gram.txt`
- the real unigram files live under `ngrams/frequency_info/`
- the defaults were updated accordingly in the builder, precompute script, and
  Oscar job wrapper

Current state right now:

- the rerun of the `mall` smoke-test has started successfully
- it created files under `ngrams/qwen_context_balanced_log_probs/`
- as of April 6, 2026 at about 9:06 PM EDT, `mall.progress.json` showed:
  - `1gram.last_line = 71000`
  - `1gram.done = false`
  - `2gram.last_line = 0`
  - `2gram.done = false`
- that means the job is past model loading and is actively scoring unigrams;
  it has not reached the selected bigram pass yet

The next step after reopening this project is:

1. Check whether the `mall` smoke-test finished cleanly.
2. If it did, launch the full array job in `oscar_jobs/precompute_qwen_focus_alt.sh`.
3. After the full precompute is complete, run `focus_alt_exp_pipeline/code/run_experiment.py --dataset qwen`.

Planned next:

- Uniform next-word baseline
- Correlation analysis between model negation probabilities and human negation probabilities on matched trials
- Finalized Qwen results and plots once the precompute has finished

## Notes For Future Contributors And AI Agents

- If you are trying to understand the current project, start in `focus_alt_exp_pipeline/code/run_experiment.py`, `focus_alt_exp_pipeline/code/runner.py`, and `focus_alt_exp_pipeline/code/models.py`.
- The active evaluation target is the human negation task, not just next-word agreement.
- The repo still contains older BERT-era work and exploratory scripts; do not assume all top-level directories reflect the current main workflow.
- The active directory is `focus_alt_exp_pipeline`.
