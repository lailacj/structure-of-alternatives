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
  Qwen preparation code.

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

Present but not fully wired into the main CLI:

- Qwen full-vocabulary continuation scoring

Planned or partially future-facing:

- Uniform next-word baseline
- More end-to-end Qwen integration

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
  --set-boundaries 2,3,4,5 \
  --num-reps 500
```

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
- If asked to integrate Qwen more fully, inspect `precompute_qwen_vocab_log_probs.py`, `run_experiment.py`, and `runner.py` together before making assumptions.

## Safe Default Mental Model

If you are unsure what the user means, assume they care most about the active focus-alternative negation pipeline in `focus_alt_exp_pipeline/`, not the archived historical analyses.
