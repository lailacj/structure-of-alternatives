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
- Alternative-structure models:
  - `ordering`
  - `set`
  - `conjunction`
  - `disjunction`
- Trial-level log-likelihood evaluation
- Plotting and context summaries

In progress:

- Qwen sparse continuation precompute over the focus-alternative contexts
- Qwen runs through the active pipeline from completed precomputed context files
- End-to-end Qwen results and plots while the full 16-context precompute is being completed

Planned or future-facing:

- Uniform next-word baseline
- Finalized project-level Qwen results once the current precompute finishes

## Qwen Handoff Status

As of April 6, 2026, the active Qwen path is the new context-balanced sparse
support pipeline, not the older global 2-gram threshold workflow.

Finished:

- `building_vocab_from_ngrams/code/build_qwen_bigram_support.py` replaces the
  old threshold-based 2-gram builder for active work
- the old threshold builders were archived under `archive/code_archive/`
- `ngrams/qwen_bigram_support/` has already been generated successfully
- every context has exactly `1500` selected bigrams
- the selected bigram union size is `22789`
- `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py` now scores
  context-selected bigrams instead of scanning the full 2-gram vocabulary
- `focus_alt_exp_pipeline/code/run_experiment.py` and
  `focus_alt_exp_pipeline/code/samplers.py` are already wired to the new
  `qwen_context_balanced_log_probs` outputs

Recent bug fix:

- the initial smoke-test failed because the precompute script expected
  `ngrams/vocab_1gram.txt`
- the actual unigram vocabulary lives at
  `ngrams/frequency_info/vocab_1gram.txt`
- the defaults and Oscar wrapper were updated to use the `frequency_info`
  unigram/count files

Current runtime state:

- a `mall` smoke-test rerun is in progress or was recently in progress under
  `ngrams/qwen_context_balanced_log_probs/`
- at the latest observed checkpoint on April 6, 2026 around 9:06 PM EDT:
  - `mall.progress.json` showed `1gram.last_line = 71000`
  - `1gram.done = false`
  - `2gram.last_line = 0`
  - `2gram.done = false`
- this indicates the sparse Qwen precompute got past model loading and path
  setup and is actively scoring unigrams

Default next step after reopening:

1. Check whether `ngrams/qwen_context_balanced_log_probs/mall.progress.json`
   now shows both sources done.
2. If `mall` completed cleanly, launch the full array job with
   `sbatch oscar_jobs/precompute_qwen_focus_alt.sh`.
3. Once all contexts are complete, run
   `focus_alt_exp_pipeline/code/run_experiment.py --dataset qwen`.

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

Qwen is still in progress, but the current code path uses:

- `focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py` for sparse context precompute
- `focus_alt_exp_pipeline/code/run_experiment.py --dataset qwen` for runs backed by completed precomputed context files
- `oscar_jobs/precompute_qwen_focus_alt.sh` for the current Oscar array-job precompute

Summarize results by context:

```bash
python focus_alt_exp_pipeline/code/summarize_log_likelihood_by_context.py \
  --results-dir focus_alt_exp_pipeline/results/cloze_probability \
  --models cloze
```

Generate plots:

```bash
python focus_alt_exp_pipeline/code/plot_results.py \
  --results-dir focus_alt_exp_pipeline/results/cloze_probability \
  --title "Cloze Probability: Average Log Likelihood by Structure"
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
- Treat cloze and frequency as stable baselines; treat Qwen as an active in-progress path until the full precompute and result set are complete.

## Safe Default Mental Model

If you are unsure what the user means, assume they care most about the active focus-alternative negation pipeline in `focus_alt_exp_pipeline/`, not the archived historical analyses.
