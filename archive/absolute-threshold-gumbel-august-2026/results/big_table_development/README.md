# Cross-dataset big-table development results

These files assemble the requested ten paper rows and are reproducible from the
canonical observations. They are development results, not yet paper-final,
because the authors still need to approve whether calibration parameters should
be shared across datasets and whether ten-fold grouped cross-validation is the
reporting rule.

Rebuild from the repository root with:

```bash
python focus_alt_exp_pipeline/code/build_cross_dataset_canonical_observations.py
python focus_alt_exp_pipeline/code/build_big_results_table.py
```

## Analysis grain

The table has four Hu benchmark rows, five Ronai & Xiang (2024) condition rows,
and one novel-focus row. Human observations are evaluated at their source
grain:

| Row family | Analysis units |
| --- | ---: |
| Hu: Ronai & Xiang (2022) | 57 scales |
| Hu: Pankratz & van Tiel (2021) | 50 scales |
| Hu: Gotzner et al. (2018) | 67 scales |
| Hu: van Tiel et al. (2016) | 39 scales, each averaging 3 sentence templates |
| Ronai & Xiang (2024) | 60 items in each of 5 conditions |
| Novel focus alternatives | 480 item types in 16 stories |

The Hu counts reproduce the literal `dropna()` subset in the public Hu
cross-scale notebook at commit
`50a7064290a841b81b2608524000b81a33ddc4b0`. In van Tiel, this excludes the
three GPT-2-unscored scales plus `unsettling/horrific`, whose unrelated LSA
field is empty. This broad source-code exclusion is preserved so every Qwen
column is evaluated on the same observations as Hu's original value.

## Current development protocol

- All model scores use summed continuation log probability from Qwen2-7B
  revision `453ed1575b739b5b03ce3758b23befdb0967f40e`.
- X-but-not-Y and no-frame columns are raw Pearson correlations. X-but-not-Y is
  defined only for Hu and novel focus.
- Structure predictions use deterministic, grouped ten-fold
  cross-validation. Hu scales, focus stories, and the five R&X condition rows
  for the same item stay in one fold.
- In each training fold, one absolute expectedness threshold and one shared
  Gumbel scale are fit using Set. Each of the ten table rows receives equal
  weight in the training objective. The fitted parameters are reused unchanged
  for Ordering, Conjunction, and Disjunction.
- Set membership is an absolute threshold event; no fixed set size is imposed,
  so expected set sizes can vary naturally by context.

The ten fits all converged, and neither the threshold nor scale reached its
search boundary. The fitted Gumbel scale ranges from about 11.24 to 12.41,
confirming that calibration is necessary before interpreting Qwen scores as
response probabilities.

## Correlations and log scores

`big_table_correlations.csv` is the compact requested correlation table.
Hu-original expectedness values are the sign-reversed Figure 3a surprisal
correlations, because greater expectedness corresponds to lower surprisal.
The four structure correlations use only held-out probabilities.

`big_table_log_scores.csv` reports two related quantities:

- `mean_item_log_score` averages
  `y log(p) + (1-y) log(1-p)` over equally weighted items. It is a proper score
  for an observed response rate and is available for every dataset.
- `response_total_log_likelihood` sums the Bernoulli trial log likelihood using
  exact yes/total counts. It omits the model-constant binomial coefficient and
  is available only for Hu Ronai-Xiang, all R&X 2024 rows, and novel focus.

The out-of-fold intercept is the training-fold mean for the corresponding
table row. `delta_item_log_score_vs_intercept` is positive when a structure
beats that baseline.

Raw Qwen X-but-not-Y and no-frame scores receive correlations only. They are
not probabilities, so a log likelihood for those columns would require a
separately specified, cross-validated calibration link.

## Files

- `analysis_units.csv`: 993 source-grain observations and fold assignments
- `oof_predictions.csv`: held-out probabilities for all four structures and
  the intercept
- `fold_parameters.csv`: training/test sizes and fitted parameters by fold
- `big_table_correlations.csv`: the requested ten-row correlation table
- `big_table_log_scores.csv`: long-form model log scores
- `big_table_metrics.csv`: wide correlation plus structure-log-score table
