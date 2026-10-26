# Four-model fit comparisons

Three grouped bar charts show No linking, X but not Y, Ordering, and Set Top-K
across the ten datasets. Each figure is provided as PNG, PDF and editable SVG.
Colors and model order are identical throughout. Higher values mean better fit;
negative log scores extend down from zero. Correlation axes share the same range.

Figures use the completed corrected single-space results. The bars are descriptive
point estimates, with no uncertainty interval or claim of statistical superiority.
R&X X-but-not-Y results are included when framed scores are supplied; missing
framed scores in frozen older bundles are labeled N/A.

Log scores sum item scores without division by item count. Compare models within
a dataset; totals depend on sample size. Log scores and Pearson correlations
are copied from the aligned dataset tables.
Hu and R&X Spearman values use the saved within-dataset/condition coefficients.
Novel-focus Spearman is the equal-weight mean of the 16 within-context negation
correlations (30 ordered pairs each), as in the current viewer. It is neither
word-ranking Spearman nor a pooled correlation over the 480 pairs.

`plotted_fit_values.csv` records every bar or unavailable slot, its aggregation,
and valid focus-context count. `provenance.json` records input hashes.

Regenerate from the repository root:

```bash
python focus_alt_exp_pipeline/code/plot_four_model_fit_comparison.py
```
