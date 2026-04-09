# AGENTS.md

This file is a practical guide for coding agents working in this repository.

## Project Summary

This repo studies whether focus-alternative generation in a human negation task can be modeled as a next-word prediction problem.

Core idea:

1. Build a next-word distribution for a context.
2. Sample orderings or sets from that distribution.
3. Turn those samples into negation probabilities for alternative-structure models.
4. Compare those predictions against human negation responses.

The main active evaluation is trial-level log likelihood. Correlation analyses between model and human negation probabilities also exist and are part of the active direction of the project.

## Where To Start

If you need to understand or modify the active pipeline, start here in this order:

1. `README.md`
2. `focus_alt_exp_pipeline/README.md`
3. `focus_alt_exp_pipeline/code/run_experiment.py`
4. `focus_alt_exp_pipeline/code/runner.py`
5. `focus_alt_exp_pipeline/code/models.py`
6. `focus_alt_exp_pipeline/code/samplers.py`

Those files define the current workflow, main abstractions, and output format.

## Active vs Inactive Areas

Treat these as the main active parts of the repo:

- `focus_alt_exp_pipeline/`
- `prompts/`
- `building_vocab_from_ngrams/`
- `next_word_prediction_correlations/` for related or supporting analyses

Treat `archive/` as reference material, not the default place to make changes, unless the task explicitly targets older work.

There is also older BERT-era and exploratory code in the repo. Do not assume older scripts reflect the current main pipeline.

## Important Repo-Specific Gotcha

Use `focus_alt_exp_pipeline` in current code, paths, and documentation.

## Active Pipeline

The current working pipeline lives in `focus_alt_exp_pipeline/`.

Main inputs:

- Human focus-alternative data:
  `focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv`
- Human cloze data:
  `focus_alt_exp_pipeline/cloze_data/all_cloze_prob_data_preprocessed.csv`

Main scripts:

- `focus_alt_exp_pipeline/code/run_experiment.py`
  Main CLI entrypoint.
- `focus_alt_exp_pipeline/code/runner.py`
  Dataset-agnostic experiment loop and result writing.
- `focus_alt_exp_pipeline/code/models.py`
  Alternative-structure models and log-likelihood conversion.
- `focus_alt_exp_pipeline/code/samplers.py`
  Next-word sampling implementations.
- `focus_alt_exp_pipeline/code/plot_results.py`
  Plot and summary generation.
- `focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py`
  Per-context summaries.
- `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py`
  Sparse Qwen precompute for focus-alternative contexts.

## Current Model Status

Implemented in the active pipeline:

- Human cloze probabilities
- Google Ngram frequency-based distributions
- Qwen next-word distributions from sparse precomputed continuation log probabilities
- Alternative-structure models:
  - `ordering`
  - `set`
  - `conjunction`
  - `disjunction`
- Trial-level log-likelihood evaluation
- Trial-level model-vs-human negation-probability correlation plots
- Plotting and context summaries

Planned or future-facing:

- Uniform next-word baseline
- Combined across-model comparison plots

## Qwen Handoff Status

As of April 9, 2026, the active Qwen path is the completed context-balanced sparse
support pipeline, not the older global 2-gram threshold workflow.

Finished:

- `building_vocab_from_ngrams/code/build_qwen_bigram_support.py` replaces the
  old threshold-based 2-gram builder for active work
- the old threshold builders were archived under `archive/code_archive/`
- `ngrams/qwen_bigram_support/` has already been generated successfully
- every context has exactly `1500` selected bigrams
- the selected bigram union size is `22789`
- `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py` now scores
  the same global-union bigram vocab for every context instead of scanning the
  full 2-gram vocabulary
- `focus_alt_exp_pipeline/code/run_experiment.py` and
  `focus_alt_exp_pipeline/code/samplers.py` are wired to the
  `qwen_context_balanced_log_probs` outputs

Recent bug fix:

- the initial smoke-test failed because the precompute script expected
  `ngrams/vocab_1gram.txt`
- the actual unigram vocabulary lives at
  `ngrams/frequency_info/vocab_1gram.txt`
- the defaults and Oscar wrapper were updated to use the `frequency_info`
  unigram/count files

Current runtime state:

- all 16 Qwen sparse precompute contexts are complete under
  `ngrams/qwen_context_balanced_log_probs/`
- the end-to-end Qwen experiment run has completed
- Qwen raw results live under `focus_alt_exp_pipeline/results/qwen/`
- Qwen plots and summary CSVs live under `focus_alt_exp_pipeline/results/qwen/plots/`

Useful current rerun path:

1. Rebuild support if needed with
   `building_vocab_from_ngrams/code/build_qwen_bigram_support.py`.
2. Re-run sparse precompute with
   `sbatch oscar_jobs/precompute_qwen_focus_alt.sh`.
3. Re-run the Qwen experiment with
   `sbatch oscar_jobs/focus_alt_exp.sh`.
4. Rebuild Qwen plots with
   `python focus_alt_exp_pipeline/code/plot_results.py --results-dir focus_alt_exp_pipeline/results/qwen --model qwen`.

## Result Conventions

Most current outputs are written under:

- `focus_alt_exp_pipeline/results/<model>/`

Common filenames:

- `ordering_results_<model>.csv`
- `set_results_<model>.csv`
- `conjunction_results_<model>.csv`
- `disjunction_results_<model>.csv`
- `missing_trials_<model>.csv`

Plots and summaries are typically written under:

- `focus_alt_exp_pipeline/results/<model>/plots/`

When changing output behavior, preserve these conventions unless the task explicitly asks for a new structure.

## Common Commands

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Run the cloze pipeline:

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

Run the Qwen experiment from completed precomputed context files:

```bash
python focus_alt_exp_pipeline/code/run_experiment.py \
  --dataset qwen \
  --qwen-top-vocab-size 100000 \
  --model-names ordering,set,conjunction,disjunction \
  --num-reps 500
```

Summarize results by context:

```bash
python focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py \
  --results-dir focus_alt_exp_pipeline/results \
  --models cloze,frequency,qwen \
  --model-order cloze,frequency,qwen
```

Generate plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/qwen \
  --model qwen
```

## Data Assumptions

Useful columns in `sca_dataframe.csv` include:

- `story`
- `trigger`
- `query`
- `neg`
- `cleaned_trigger`
- `cleaned_query`

Useful columns in the cloze dataset include:

- `context`
- `word`
- `cloze_probability`

When changing data loading or preprocessing logic, prefer keeping compatibility with the cleaned columns already used by the current runner.

## Guidance For Agents

- Prefer making changes inside the active pipeline unless the task explicitly concerns older analyses.
- Read both READMEs before refactoring behavior that affects CLI usage, outputs, or project framing.
- Preserve existing result filenames and directory layout where possible.
- Be careful with long-running scripts or large generated outputs; this repo contains many result artifacts already.
- If adding documentation, keep it aligned with the current active workflow rather than archival code.
- If asked to work on Qwen, inspect `precompute_qwen_vocab_log_probs.py`, `samplers.py`, `run_experiment.py`, and `runner.py` together before making assumptions.
- Treat cloze, frequency, and Qwen as implemented active model paths in the main pipeline.

## Safe Default Mental Model

If you are unsure what the user means, assume they care most about the active focus-alternative negation pipeline in `focus_alt_exp_pipeline/`, not the archived historical analyses.
