# Trigger Analysis

This folder contains a standalone Qwen analysis for comparing query continuation
log probabilities under two prompt types:

1. Base next-word prompt:
   `log P_Qwen(query | base_prefix)`
2. Trigger-negated prompt:
   `log P_Qwen(query | base_prefix + trigger + " but not")`

The analysis uses unique `(story, trigger, query)` pairs from the human
focus-alternative experiment and prompt prefixes from
`prompts/prompt_files/prompts_llm_next_word.csv`.

## Run

From the repo root:

```bash
python trigger_analysis/code/score_qwen_trigger_query_correlations.py \
  --local-files-only
```

The default Qwen model path is:

```text
../hf-cache/models--Qwen--Qwen2-7B
```

Use `--dry-run` to check the rows and prompt groups without loading Qwen:

```bash
python trigger_analysis/code/score_qwen_trigger_query_correlations.py --dry-run
```

## Outputs

The script writes to `trigger_analysis/results/` by default:

- `qwen_trigger_query_pair_logprobs.csv`
  Trial-type scores for unique `(story, trigger, query)` pairs.
- `qwen_trigger_query_pair_correlations.csv`
  Pooled and per-context pair-level correlations.
- `qwen_unique_query_logprobs.csv`
  Unique `(story, query)` rows, with trigger-conditioned logprobs aggregated
  across the five triggers paired with that query.
- `qwen_unique_query_correlations.csv`
  Pooled and per-context unique-query correlations.

The primary score is summed continuation log probability. Mean per-token log
probability is also included as a length sensitivity check for multiword queries.

## Plots

After scoring, generate the three recommended presentation plots with:

```bash
python trigger_analysis/code/plot_qwen_trigger_query_results.py
```

By default this writes PNG and PDF files under `trigger_analysis/results/plots/`:

- unique-query base-vs-trigger scatter plot
- per-context unique-query correlation dot plot
- selected-context trigger-by-query delta heatmap panel
- one individual trigger-by-query delta heatmap per context under
  `trigger_analysis/results/plots/context_heatmaps/`

To plot mean token log probabilities instead of summed log probabilities:

```bash
python trigger_analysis/code/plot_qwen_trigger_query_results.py \
  --score-scale mean_logprob_per_token
```

## Note On Unique Query Correlations

For a fixed `(story, query)`, the base log probability is constant across all
trigger prompts. So the script does not compute a within-query correlation across
triggers. Instead, it aggregates trigger-conditioned log probabilities for each
unique query, then correlates base log probability with the aggregated
trigger-conditioned value across unique queries.
